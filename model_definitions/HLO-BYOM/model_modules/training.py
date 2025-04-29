from teradataml import *
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

def train(context: ModelContext, **kwargs):
    aoa_create_context()
    configure.val_install_location = os.environ.get("AOA_VAL_INSTALL_DB", "VAL") 
    #configure.val_install_location = 'VAL'
    
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    #feature_names_str = ",".join(["'{}'".format(value) for value in feature_names])
    feature_summary = get_feature_stats_summary(context.dataset_info.get_feature_metadata_fqtn())
    categorical_features = [f for f in feature_names if feature_summary[f.lower()] == 'categorical']

    # Read training dataset from Teradata
    train_df = DataFrame.from_query(context.dataset_info.sql)
    
    print("Loading Model into Vantage...")

    save_byom(model_id=f"{context.model_id}", 
             model_file=f"{context.artifact_output_path}/model.onnx", 
             table_name="BYOM_Models",
             additional_columns={"model_version": f"{context.model_version}", 
                                 "model_type": "ONNX", 
                                 "project_id": f"{context.project_id}",
                                 "deployed_at": datetime.datetime.now()})

    print("Model Loaded")
        
    # export model artifacts
    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=categorical_features + [target_name],
                          context=context)
