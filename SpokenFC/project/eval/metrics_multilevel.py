import argparse
import json
import re
import ast
import os
from typing import List, Dict, Tuple, Set, Any, Optional

# --- Text normalization ---

try:
    from opencc import OpenCC
except ImportError:
    print("Error: missing package 'opencc-python-reimplemented'.")
    print("Install it with: pip install opencc-python-reimplemented")
    exit(1)

try:
    import regex as regex_module
except ImportError:
    print("Error: missing package 'regex'.")
    print("Install it with: pip install regex")
    exit(1)


cc = OpenCC('t2s')
# Chinese and English number word -> digit mapping for normalization
NUM_MAP = {
    '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}

def normalize_text(text: str) -> str:
    """
    Normalized text for metric comparison:
    - Traditional to simplified Chinese
    - Lowercase letters (all languages)
    - Remove the word 'the'
    - Map Chinese/English number words to Arabic digits
    - Remove whitespace and punctuation except ':', keep letters and digits
    """
    if not isinstance(text, str):
        return ""
    
    text = cc.convert(text)
    text = text.lower()
    text = re.sub(r'\bthe\b', '', text)
    for word, digit in NUM_MAP.items():
        text = text.replace(word, digit)
    text = regex_module.sub(r'[^\p{L}\p{N}:]', '', text)
    return text

# --- Parsers and normalization helpers ---

SFC_PATTERN = re.compile(r"(\w+)\s*\((.*)\)")
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL) 

NormalizedCall = Tuple[str, Dict[str, str]]

def parse_sfc_string(sfc_str: str) -> Optional[NormalizedCall]:
    sfc_str = sfc_str.strip()
    match = SFC_PATTERN.match(sfc_str)
    if not match: return None
    tool_name, args_str = match.group(1), match.group(2)
    if args_str.endswith(')'): args_str = args_str[:-1]
    slots = {}
    try:
        temp_args_str = args_str.strip()
        while temp_args_str:
            key_match = re.match(r'^\s*(\w+)\s*=\s*"', temp_args_str)
            if not key_match: break
            key = key_match.group(1)
            temp_args_str = temp_args_str[key_match.end():]
            val = ""
            i = 0
            while i < len(temp_args_str):
                if temp_args_str[i] == '"' and (i == 0 or temp_args_str[i-1] != '\\'): break
                val += temp_args_str[i]
                i += 1
            slots[key] = val.replace('\\"', '"')
            temp_args_str = temp_args_str[i+1:].lstrip(' ,')
    except Exception:
        return (tool_name, {})
    return (tool_name, slots)

def robust_parser(content_str: str, format_type: str) -> Tuple[List[Any], Optional[Exception]]:
    if not content_str or content_str in ('[]', '""', "''"): 
        return [], None
    try:
        if format_type == 'sfc':
            try:
                data = ast.literal_eval(content_str)
                if isinstance(data, list): 
                    return data, None
            except (SyntaxError, ValueError):
                pass
            sfc_strings = re.findall(r'\w+\s*\(.*?\)', content_str)
            if sfc_strings: 
                return sfc_strings, None
            raise ValueError("SFC string is unparsable by both ast.literal_eval and regex.")
        else: # for 'slu'
            data = ast.literal_eval(content_str)
            if not isinstance(data, list):
                raise ValueError("SLU format expects a list of dictionaries.")
            return data, None
    except Exception as e:
        return [], e

def normalize_slu(data: List[Dict[str, Any]]) -> List[NormalizedCall]:
    normalized_list = []
    if not isinstance(data, list): return []
    for item in data:
        if not isinstance(item, dict): continue
        intent = item.get("intent", "") or item.get("domain", "")
        slots = item.get("slots", {})
        if not isinstance(slots, dict): slots = {}
        normalized_list.append((intent, slots))
    return normalized_list

def normalize_sfc(data: List[Any]) -> List[NormalizedCall]:
    normalized_list = []
    if not data: return []
    if (len(data) == 2 and isinstance(data[0], str) and isinstance(data[1], dict)):
        tool_name = data[0]
        slots = {k: str(v) for k, v in data[1].items()}
        normalized_list.append((tool_name, slots))
        return normalized_list
    for item in data:
        if isinstance(item, str):
            parsed = parse_sfc_string(item)
            if parsed: normalized_list.append(parsed)
    return normalized_list

# --- Metric computation ---

def get_canonical_representation(calls: List[NormalizedCall]) -> Set[Tuple[str, frozenset]]:
    canonical_set = set()
    for intent, slots in calls:
        normalized_intent = normalize_text(intent)
        normalized_slots = frozenset(
            (normalize_text(k), normalize_text(str(v))) for k, v in slots.items()
        )
        canonical_set.add((normalized_intent, normalized_slots))
    return canonical_set

def calculate_metrics(predictions: List[List[NormalizedCall]], labels: List[List[NormalizedCall]]):
    total_samples = len(predictions)
    if total_samples == 0:
        return {
            "intent_accuracy": 0, "slot_precision": 0, "slot_recall": 0,
            "slot_f1": 0, "overall_accuracy": 0, "total_samples": 0
        }
    correct_intent_count, correct_overall_count = 0, 0
    total_slot_tp, total_slot_fp, total_slot_fn = 0, 0, 0
    for pred_calls, label_calls in zip(predictions, labels):
        pred_canonical = get_canonical_representation(pred_calls)
        label_canonical = get_canonical_representation(label_calls)
        
        if pred_canonical == label_canonical:
            correct_overall_count += 1
            
        pred_intents = {normalize_text(intent) for intent, _ in pred_calls}
        label_intents = {normalize_text(intent) for intent, _ in label_calls}
        if pred_intents == label_intents:
            correct_intent_count += 1
            
        pred_slots, label_slots = set(), set()
        for _, slots in pred_calls:
            pred_slots.update((normalize_text(k), normalize_text(str(v))) for k, v in slots.items())
        for _, slots in label_calls:
            label_slots.update((normalize_text(k), normalize_text(str(v))) for k, v in slots.items())
            
        total_slot_tp += len(pred_slots.intersection(label_slots))
        total_slot_fp += len(pred_slots - label_slots)
        total_slot_fn += len(label_slots - pred_slots)
        
    intent_accuracy = correct_intent_count / total_samples if total_samples > 0 else 0
    overall_accuracy = correct_overall_count / total_samples if total_samples > 0 else 0
    slot_precision = total_slot_tp / (total_slot_tp + total_slot_fp) if (total_slot_tp + total_slot_fp) > 0 else 0
    slot_recall = total_slot_tp / (total_slot_tp + total_slot_fn) if (total_slot_tp + total_slot_fn) > 0 else 0
    slot_f1 = 2 * (slot_precision * slot_recall) / (slot_precision + slot_recall) if (slot_precision + slot_recall) > 0 else 0
    
    return {
        "intent_accuracy": intent_accuracy, "slot_precision": slot_precision,
        "slot_recall": slot_recall, "slot_f1": slot_f1,
        "overall_accuracy": overall_accuracy, "total_samples": total_samples
    }

# --- Level helpers ---

ID_STAGE_PATTERN = re.compile(r"(.+)_([12])\.(wav|flac|opus|m4a|mp3|pcm)$", re.IGNORECASE)

def get_level_info(audio_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not audio_path or not isinstance(audio_path, str):
        return None, None, None
    level = None
    if "/level_1/" in audio_path: level = "L1"
    elif "/level_2/" in audio_path: level = "L2"
    elif "/level_3_1/" in audio_path: level = "L3_1"
    elif "/level_3_2/" in audio_path: level = "L3_2"
    if level not in ["L3_1", "L3_2"]:
        return level, None, None 
    filename = os.path.basename(audio_path)
    match = ID_STAGE_PATTERN.search(filename)
    if match:
        base_id = match.group(1)
        stage = match.group(2)
        return level, base_id, stage
    return level, None, None

def calculate_metrics_for_level(data_pairs: List[Tuple[List[NormalizedCall], List[NormalizedCall]]]):
    predictions = [pair[0] for pair in data_pairs]
    labels = [pair[1] for pair in data_pairs]
    return calculate_metrics(predictions, labels)

# --- Level 3 metrics ---

def calculate_l3_metrics(
    l3_s1_results: Dict[str, Dict], 
    l3_s2_results: Dict[str, Dict]
) -> Dict[str, Any]:
    total_units, correct_intent_units, correct_overall_units = 0, 0, 0
    total_slot_tp, total_slot_fp, total_slot_fn = 0, 0, 0
    orphans_s1, processed_s2_ids = 0, set() 

    for base_id, stage1 in l3_s1_results.items():
        stage2 = l3_s2_results.get(base_id)
        if stage2:
            total_units += 1
            processed_s2_ids.add(base_id)
            if stage1["intent_correct"] and stage2["intent_correct"]: correct_intent_units += 1
            if stage1["overall_correct"] and stage2["overall_correct"]: correct_overall_units += 1
            total_slot_tp += stage1["tp"] + stage2["tp"]
            total_slot_fp += stage1["fp"] + stage2["fp"]
            total_slot_fn += stage1["fn"] + stage2["fn"]
        else:
            orphans_s1 += 1
    orphans_s2 = len(set(l3_s2_results.keys()) - processed_s2_ids)

    intent_accuracy = correct_intent_units / total_units if total_units > 0 else 0
    overall_accuracy = correct_overall_units / total_units if total_units > 0 else 0
    slot_precision = total_slot_tp / (total_slot_tp + total_slot_fp) if (total_slot_tp + total_slot_fp) > 0 else 0
    slot_recall = total_slot_tp / (total_slot_tp + total_slot_fn) if (total_slot_tp + total_slot_fn) > 0 else 0
    slot_f1 = 2 * (slot_precision * slot_recall) / (slot_precision + slot_recall) if (slot_precision + slot_recall) > 0 else 0

    return {
        "intent_accuracy": intent_accuracy, "slot_precision": slot_precision,
        "slot_recall": slot_recall, "slot_f1": slot_f1,
        "overall_accuracy": overall_accuracy, "total_samples": total_units,
        "orphans_s1": orphans_s1, "orphans_s2": orphans_s2
    }

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Robustly calculate accuracy for SLU/SFC tasks with text normalization.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--format", type=str, choices=["slu", "sfc"], required=True, help="The format of the prediction/label data.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output JSONL file to save detailed results.")
    parser.add_argument("--verbose", action="store_true", help="Print warnings for each unparsable field with detailed error.")
    args = parser.parse_args()

    all_predictions, all_labels = [], []
    level_data_l1: List[Tuple[List[NormalizedCall], List[NormalizedCall]]] = []
    level_data_l2: List[Tuple[List[NormalizedCall], List[NormalizedCall]]] = []
    l3_1_stage1_results: Dict[str, Dict] = {}
    l3_1_stage2_results: Dict[str, Dict] = {}
    l3_2_stage1_results: Dict[str, Dict] = {}
    l3_2_stage2_results: Dict[str, Dict] = {}
    detailed_results = []
    skipped_line_count, unparsed_field_count = 0, 0
    unparsed_line_numbers = []
    unbucketed_l3_count = 0 

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    if args.verbose: print(f"Warning: Skipping line #{i} due to invalid JSON structure.")
                    skipped_line_count += 1
                    continue
                
                pred_str = data.get('predict', data.get('prediction', ''))
                label_str = data.get('label', '')
                original_pred_str = pred_str 
                cleaned_pred_str = THINK_PATTERN.sub("", pred_str).strip() 
                cleaned_label_str = THINK_PATTERN.sub("", label_str).strip() 
                raw_pred, pred_error = robust_parser(cleaned_pred_str, args.format)
                raw_label, label_error = robust_parser(cleaned_label_str, args.format) 

                if pred_error:
                    unparsed_field_count += 1
                    if i not in unparsed_line_numbers: unparsed_line_numbers.append(i)
                    if args.verbose: print(f"Warning: Unparsable prediction on line #{i}, treated as empty. Error: {pred_error}.")
                if label_error:
                    unparsed_field_count += 1
                    if i not in unparsed_line_numbers: unparsed_line_numbers.append(i)
                    if args.verbose: print(f"Warning: Unparsable label on line #{i}, treated as empty. Error: {label_error}.")

                if args.format == 'slu':
                    norm_pred, norm_label = normalize_slu(raw_pred), normalize_slu(raw_label)
                else:
                    norm_pred, norm_label = normalize_sfc(raw_pred), normalize_sfc(raw_label)
                
                pred_canonical = get_canonical_representation(norm_pred)
                label_canonical = get_canonical_representation(norm_label)
                is_sample_correct = pred_canonical == label_canonical
                
                pred_intents = {normalize_text(intent) for intent, _ in norm_pred}
                label_intents = {normalize_text(intent) for intent, _ in norm_label}
                is_intent_correct = pred_intents == label_intents
                
                pred_slots, label_slots = set(), set()
                for _, slots in norm_pred:
                    pred_slots.update((normalize_text(k), normalize_text(str(v))) for k, v in slots.items())
                for _, slots in norm_label:
                    label_slots.update((normalize_text(k), normalize_text(str(v))) for k, v in slots.items())
                
                sample_tp = len(pred_slots.intersection(label_slots))
                sample_fp = len(pred_slots - label_slots)
                sample_fn = len(label_slots - pred_slots)

                data['predict'] = original_pred_str 
                data['label'] = label_str       
                data['is_correct'] = is_sample_correct
                
                audio_path = data.get("audio_path", "")
                if 'audio_path' not in data: data['audio_path'] = ''
                    
                if args.output_file:
                    detailed_results.append(data)
                
                all_predictions.append(norm_pred)
                all_labels.append(norm_label)

                level, base_id, stage = get_level_info(audio_path)
                
                if level == "L1":
                    level_data_l1.append((norm_pred, norm_label))
                elif level == "L2":
                    level_data_l2.append((norm_pred, norm_label))
                elif level == "L3_1":
                    if base_id and stage == "1":
                        l3_1_stage1_results[base_id] = {"intent_correct": is_intent_correct, "overall_correct": is_sample_correct, "tp": sample_tp, "fp": sample_fp, "fn": sample_fn}
                    elif base_id and stage == "2":
                        l3_1_stage2_results[base_id] = {"intent_correct": is_intent_correct, "overall_correct": is_sample_correct, "tp": sample_tp, "fp": sample_fp, "fn": sample_fn}
                    else:
                        unbucketed_l3_count += 1
                elif level == "L3_2":
                    if base_id and stage == "1":
                        l3_2_stage1_results[base_id] = {"intent_correct": is_intent_correct, "overall_correct": is_sample_correct, "tp": sample_tp, "fp": sample_fp, "fn": sample_fn}
                    elif base_id and stage == "2":
                        l3_2_stage2_results[base_id] = {"intent_correct": is_intent_correct, "overall_correct": is_sample_correct, "tp": sample_tp, "fp": sample_fp, "fn": sample_fn}
                    else:
                        unbucketed_l3_count += 1
                
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        return

    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f_out:
                for item in detailed_results:
                    clean_item = { "prompt": item.get("prompt", ""), "predict": item.get("predict", ""), "label": item.get("label", ""), "is_correct": item.get("is_correct", False), "audio_path": item.get("audio_path", "") }
                    f_out.write(json.dumps(clean_item, ensure_ascii=False) + '\n')
            print(f"\nDetailed results with 'is_correct' field saved to: {args.output_file}")
        except Exception as e:
            print(f"\nError writing to output file '{args.output_file}': {e}")

    results_total = calculate_metrics(all_predictions, all_labels)
    results_l1 = calculate_metrics_for_level(level_data_l1)
    results_l2 = calculate_metrics_for_level(level_data_l2)
    results_l3_1 = calculate_l3_metrics(l3_1_stage1_results, l3_1_stage2_results)
    results_l3_2 = calculate_l3_metrics(l3_2_stage1_results, l3_2_stage2_results)

    print("\n" + "="*85)
    print(f"Evaluation Results for: {args.input_file}")
    print(f"Format: {args.format.upper()}")
    print("-" * 85)
    print(f"Total Samples Processed: {results_total['total_samples']}")
    
    if unparsed_field_count > 0: 
        print(f"Unparsable 'predict'/'label' fields: {unparsed_field_count} (treated as errors)")
        if unparsed_line_numbers:
            max_lines_to_print = 20
            lines_str = ', '.join(map(str, unparsed_line_numbers[:max_lines_to_print]))
            if len(unparsed_line_numbers) > max_lines_to_print: lines_str += f", ... (and {len(unparsed_line_numbers) - max_lines_to_print} more)"
            print(f"  -> Occurred on lines: [{lines_str}]")
    if skipped_line_count > 0: print(f"Skipped Lines (invalid JSON structure): {skipped_line_count}")
    if unbucketed_l3_count > 0: print(f"L3 Warning: {unbucketed_l3_count} L3_1/L3_2 samples failed ID/Stage parsing and were excluded from L3 metrics.")
    if results_l3_1["orphans_s1"] > 0 or results_l3_1["orphans_s2"] > 0:
        print(f"L3_1 Warning: Found incomplete units (orphans).")
        if results_l3_1["orphans_s1"] > 0: print(f"  - L3_1 Stage 1 only: {results_l3_1['orphans_s1']}")
        if results_l3_1["orphans_s2"] > 0: print(f"  - L3_1 Stage 2 only: {results_l3_1['orphans_s2']}")
    if results_l3_2["orphans_s1"] > 0 or results_l3_2["orphans_s2"] > 0:
        print(f"L3_2 Warning: Found incomplete units (orphans).")
        if results_l3_2["orphans_s1"] > 0: print(f"  - L3_2 Stage 1 only: {results_l3_2['orphans_s1']}")
        if results_l3_2["orphans_s2"] > 0: print(f"  - L3_2 Stage 2 only: {results_l3_2['orphans_s2']}")
        
    print("="*85)
    print("Metrics are calculated AFTER applying text normalization")
    print("L3_1/L3_2 Tool/Overall Acc requires BOTH Stage 1 AND Stage 2 in that category to be correct.")
    print("-" * 85)
    
    print(f"{'Metric':<14} | {'Level 1':<12} | {'Level 2':<12} | {'L3_1 (Units)':<14} | {'L3_2 (Units)':<14} | {'OVERALL':<12}")
    print(f"{'-'*14:14} | {'-'*12:12} | {'-'*12:12} | {'-'*14:14} | {'-'*14:14} | {'-'*12:12}")

    def print_metric_row(metric_name: str, key: str):
        l1_val = f"{results_l1[key]:.4f}" if results_l1['total_samples'] > 0 else "N/A"
        l2_val = f"{results_l2[key]:.4f}" if results_l2['total_samples'] > 0 else "N/A"
        l3_1_val = f"{results_l3_1[key]:.4f}" if results_l3_1['total_samples'] > 0 else "N/A"
        l3_2_val = f"{results_l3_2[key]:.4f}" if results_l3_2['total_samples'] > 0 else "N/A"
        total_val = f"{results_total[key]:.4f}" if results_total['total_samples'] > 0 else "N/A"
        print(f"{metric_name:<14} | {l1_val:>12} | {l2_val:>12} | {l3_1_val:>14} | {l3_2_val:>14} | {total_val:>12}")
        
    def print_count_row(metric_name: str, key: str):
        l1_val = results_l1[key]
        l2_val = results_l2[key]
        l3_1_val = results_l3_1[key]
        l3_2_val = results_l3_2[key]
        total_val = results_total[key]
        print(f"{metric_name:<14} | {l1_val:>12} | {l2_val:>12} | {l3_1_val:>14} | {l3_2_val:>14} | {total_val:>12}")

    print_metric_row("Tool Name Acc", "intent_accuracy")
    print_metric_row("Overall Acc", "overall_accuracy")
    print("-" * 85)
    print("Param Value (Slots):")
    print_metric_row("  - Precision", "slot_precision")
    print_metric_row("  - Recall", "slot_recall")
    print_metric_row("  - F1 Score", "slot_f1")
    print("-" * 85)
    print_count_row("Sample Count", "total_samples")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
