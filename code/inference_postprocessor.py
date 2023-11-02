

import os
import numpy as np
import pandas as pd
import argparse
import json
import tarfile
import joblib

from pathlib import Path
from io import StringIO

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from pickle import dump, load
from sagemaker_containers.beta.framework import encoders, worker



def input_fn(input_data, content_type):
    if content_type == "application/json":
        predictions = json.loads(input_data)["predictions"]
        return predictions
    
    else:
        raise ValueError(f"{content_type} is not supported.!")


def output_fn(prediction, accept):
    if accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    
    if accept == "application/json":
        response = []
        for p, c in prediction:
            response.append({
                "prediction": p,
                "confidence": c
            })

        # If there's only one prediction, we'll return it as a single object.
        if len(response) == 1:
            response = response[0]
            
        return worker.Response(json.dumps(response), mimetype=accept)
    
    raise RuntimeException(f"{accept} accept type is not supported.")


def predict_fn(input_data, model):
    """
    Transforms the prediction into its corresponding category.
    """

    predictions = np.argmax(input_data, axis=-1)
    confidence = np.max(input_data, axis=-1)
    return [(model[prediction], confidence) for confidence, prediction in zip(confidence, predictions)]


def model_fn(model_dir):
    """
    Deserializes the target model and returns the list of fitted categories.
    """
    
    model = joblib.load(os.path.join(model_dir, "target.joblib"))
    return model.named_transformers_["species"].categories_[0]
