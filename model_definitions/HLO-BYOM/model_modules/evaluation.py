from teradataml import *
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
from sklearn import metrics
import json

def evaluate(context: ModelContext, **kwargs):
    aoa_create_context()
    configure.val_install_location = os.environ.get("AOA_VAL_INSTALL_DB", "VAL") 
    configure.byom_install_location = os.environ.get("AOA_BYOM_INSTALL_DB", "MLDB")
    
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    test_df = DataFrame.from_query(context.dataset_info.sql)

    print("Loading Model")
    
    model = DataFrame.from_query(f"""
       SELECT model_version as model_id, model
        FROM BYOM_Models WHERE model_version='{context.model_version}'
       """)
    
    print("Evaluating")
    
    res = ONNXPredict(
            modeldata = model,
            newdata = test_df,
            accumulate = [target_name],
            model_input_fields_map = 'float_input=' + ','.join(feature_names)
        ).result

    predictions_df = DataFrame.from_query(f'''
    Select {target_name}, CAST(NEW JSON(json_report, LATIN).JSONEXTRACTValue('$.label.0') AS BIGINT) as Prediction 
    from {res.show_query().split(".")[1]};''')
    
    print("Finished Evaluating")

    eval_stats = ClassificationEvaluator(data=predictions_df, observation_column=target_name, prediction_column='Prediction', labels=['0','1'])
    eval_data = eval_stats.output_data.to_pandas().reset_index(drop=True)

    evaluation = {
        'Accuracy': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Accuracy')].MetricValue.item()),
        'Recall': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Macro-Recall')].MetricValue.item()),
        'Precision': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Macro-Precision')].MetricValue.item()),
        'f1-score': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Macro-F1')].MetricValue.item())
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)
        
    # Confusion Matrix sklearn metrics
    predictions_pdf = predictions_df.to_pandas()
    cm = metrics.confusion_matrix(predictions_pdf[target_name],predictions_pdf["Prediction"])
    cmd = metrics.ConfusionMatrixDisplay(cm,display_labels = ['0','1'])
    cmd.plot()
    save_plot('Confusion Matrix', context=context)
    
    fpr, tpr, thresholds = metrics.roc_curve(predictions_pdf[target_name],predictions_pdf["Prediction"] , pos_label=1)
    roc_AUC = metrics.auc(fpr,tpr)
    rcd = metrics.RocCurveDisplay(fpr = fpr, tpr = tpr,roc_auc = roc_AUC, estimator_name = 'HLO_BYOM')
    
    rcd.plot()
    save_plot('ROC Curve', context=context)
    
    record_evaluation_stats(features_df=test_df,
                            predicted_df=predictions_df,
                            context=context)