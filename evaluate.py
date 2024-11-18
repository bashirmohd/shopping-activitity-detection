import pandas as pd
import argparse
parser = argparse.ArgumentParser(
    description='Evaluate method accuracy.'
)
parser.add_argument(
    '--csv', 
    type=str, 
    required=True,
    help='Path to the CSV file containing the analysis results'
)
parser.add_argument(
    '--ground-truth', 
    type=str, 
    default='ground_truths.csv',
    help='Path to the CSV file containing ground truth labels (default: ground_truths.csv)'
)
args = parser.parse_args()

try:
    # Read both CSVs
    df = pd.read_csv(args.csv)
    ground_truth_df = pd.read_csv(args.ground_truth)
    
    # Merge results with ground truth
    df = df.merge(ground_truth_df, on='filename', how='left', suffixes=('', '_truth'))
    
    # Rename the ground truth column to avoid confusion
    df = df.rename(columns={'anomaly_truth': 'ground_truth'})
    
    # Check for any videos without ground truth
    missing_truth = df[df['ground_truth'].isna()]
    if len(missing_truth) > 0:
        print("\nWarning: Some videos missing ground truth:")
        for filename in missing_truth['filename']:
            print(f"- {filename}")
        
        # Remove entries with missing ground truth
        df = df.dropna(subset=['ground_truth'])
    
    valid_results = df[df['anomaly'] != 'ERROR']
    error_cases = df[df['anomaly'] == 'ERROR']
    
    total_videos = len(valid_results)
    detected_anomalies = len(valid_results[valid_results['anomaly'] == 'Yes'])
    actual_anomalies = len(valid_results[valid_results['ground_truth'] == 'Yes'])
    
    # Calculate true positives, false positives, true negatives, false negatives
    tp = len(valid_results[(valid_results['anomaly'] == 'Yes') & (valid_results['ground_truth'] == 'Yes')])
    fp = len(valid_results[(valid_results['anomaly'] == 'Yes') & (valid_results['ground_truth'] == 'No')])
    tn = len(valid_results[(valid_results['anomaly'] == 'No') & (valid_results['ground_truth'] == 'No')])
    fn = len(valid_results[(valid_results['anomaly'] == 'No') & (valid_results['ground_truth'] == 'Yes')])
    
    metrics = {
        'total_processed': len(df),
        'total_valid': total_videos,
        'total_errors': len(error_cases),
        'detected_anomalies': detected_anomalies,
        'actual_anomalies': actual_anomalies,
        'accuracy': (tp + tn) / total_videos if total_videos > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'error_rate': len(error_cases) / len(df)
    }
    
    print("\n=== Anomaly Detection Accuracy Report ===")
    print(f"\nTotal Videos Processed: {metrics['total_processed']}")
    print(f"Valid Results: {metrics['total_valid']}")
    print(f"Processing Errors: {metrics['total_errors']}")
    print(f"\nDetected Anomalies: {metrics['detected_anomalies']}")
    print(f"Actual Anomalies: {metrics['actual_anomalies']}")
    
    print("\nPerformance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"Error Rate: {metrics['error_rate']:.2%}")
    
    print("\nDetailed Analysis:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Print details of misclassified videos
    misclassified = valid_results[valid_results['anomaly'] != valid_results['ground_truth']]
    if len(misclassified) > 0:
        print("\nMisclassified Videos:")
        for idx, row in misclassified.iterrows():
            print(f"- {row['filename']}")
            print(f"  Predicted: {row['anomaly']}")
            print(f"  Actual: {row['ground_truth']}")
            print(f"  Reason given: {row['reason']}")
            if 'duration' in row:
                print(f"  Duration: {row['duration']:.1f}s")
                if row['trimmed']:
                    print("  Note: Video was trimmed to 30s")
            print()
    
    # Print details of error cases
    if len(error_cases) > 0:
        print("\nProcessing Errors:")
        for idx, row in error_cases.iterrows():
            print(f"- {row['filename']}")
            print(f"  Error: {row['reason']}")
            print()

except FileNotFoundError as e:
    print(f"Error: Could not find file - {str(e)}")
except Exception as e:
    print(f"Error analyzing results: {str(e)}")