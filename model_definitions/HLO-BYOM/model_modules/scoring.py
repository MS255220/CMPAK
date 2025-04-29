from teradataml import copy_to_sql, DataFrame
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

import os

def score(context: ModelContext, **kwargs):
    aoa_create_context()

    configure.val_install_location = os.environ.get("AOA_VAL_INSTALL_DB", "VAL") 
    configure.byom_install_location = os.environ.get("AOA_BYOM_INSTALL_DB", "MLDB")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    features_df = DataFrame.from_query(context.dataset_info.sql)
    
    print("Scoring")
    
    model = DataFrame.from_query(f"""
       SELECT model_version as model_id, model
        FROM BYOM_MODELS WHERE model_version='{context.model_version}'
       """)
    
    res = ONNXPredict(
            modeldata = model,
            newdata = features_df,
            accumulate = [entity_key],
            model_input_fields_map = 'float_input=' + ','.join(feature_names)
        ).result

    predictions_df = DataFrame.from_query(f'''
    Select {entity_key}, CAST(NEW JSON(json_report, LATIN).JSONEXTRACTValue('$.label.0') AS BIGINT) as {target_name}, json_report
    from {res.show_query().split(".")[1]};''')
    
    print("Finished Scoring")

    # store the predictions
    # add job_id column so we know which execution this is from if appended to predictions table
    predictions_df = predictions_df.assign(job_id=context.job_id)

    predictions_df[['job_id', entity_key, target_name, 'json_report']].to_sql(
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                if_exists="append")
    
    print("Saved predictions in Teradata")

    # calculate stats of this job
    record_scoring_stats(features_df=features_df, 
                         predicted_df=predictions_df, 
                         context=context)