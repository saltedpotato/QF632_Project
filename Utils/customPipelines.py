from sklearn.pipeline import Pipeline
from Utils.pipelineComponents import *
import warnings

warnings.filterwarnings("ignore")

def get_pipeline_clean_encode_only(categorical_features_one_hot=[], categorical_features_ordinal=[], numerical_features=[]):
  # Clean data with categorical encoding
  categorical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=categorical_features_one_hot + categorical_features_ordinal)),
    ('categorical_imputer', CustomImputer(impute_type='categorical')),
    ('encoder', CustomEncoder(categorical_features_one_hot, categorical_features_ordinal)),
  ])
  numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=numerical_features))
  ])
  preprocessed_pipeline = PandasFeatureUnion([
    ('categorical_pipeline', categorical_pipeline), 
    ('numerical_pipeline', numerical_pipeline),
  ])
  return preprocessed_pipeline


def get_pipeline_clean_encode_impute(categorical_features_one_hot=[], categorical_features_ordinal=[], numerical_features=[]):
  categorical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=categorical_features_one_hot + categorical_features_ordinal)),
    ('categorical_imputer', CustomImputer(impute_type='categorical')),
    ('encoder', CustomEncoder(categorical_features_one_hot, categorical_features_ordinal)),
  ])
  numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=numerical_features)),
    ('numerical_imputer', CustomImputer(impute_type='numerical'))
  ])
  preprocessed_pipeline = PandasFeatureUnion([
    ('categorical_pipeline', categorical_pipeline), 
    ('numerical_pipeline', numerical_pipeline), 
  ])
  return preprocessed_pipeline


def get_pipeline_clean_encode_outlier_impute(categorical_features_one_hot=[], categorical_features_ordinal=[], numerical_features=[], outlier_numerical_features=[]):
  categorical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=categorical_features_one_hot + categorical_features_ordinal)),
    ('categorical_imputer', CustomImputer(impute_type='categorical')),
    ('encoder', CustomEncoder(categorical_features_one_hot, categorical_features_ordinal)),
  ])
  numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=numerical_features)),
    ('numerical_imputer', CustomImputer(impute_type='numerical'))
  ])
  outlier_numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=outlier_numerical_features)),
    ("outlier_handler", OutlierHandler(outlier_algorithm="IsolationForest", verbose=False)), # Mark outlier values as NaN
    ('numerical_imputer', CustomImputer(impute_type='numerical'))
  ])

  preprocessed_pipeline = PandasFeatureUnion([
    ('categorical_pipeline', categorical_pipeline),
    ('numerical_pipeline', numerical_pipeline), 
    ('outlier_numerical_pipeline', outlier_numerical_pipeline)
  ])
  return preprocessed_pipeline
