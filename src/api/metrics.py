"""Prometheus metrics exposed on /metrics."""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("churn_http_requests_total", "HTTP requests",
                        ["method", "path", "status"])
HTTP_LATENCY = Histogram("churn_http_request_duration_seconds", "HTTP latency (s)",
                         ["method", "path"],
                         buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
PREDICTIONS = Counter("churn_predictions_total", "Predictions by outcome", ["outcome"])
PROBABILITY = Histogram("churn_prediction_probability", "Predicted churn probability",
                        buckets=[i / 10 for i in range(1, 11)])
PREDICTION_ERRORS = Counter("churn_prediction_errors_total", "Inference failures")
MODEL_INFO = Gauge("churn_model_info", "Loaded model (value is always 1)",
                   ["version", "model_type"])
MODEL_AUC = Gauge("churn_model_roc_auc", "Hold-out ROC-AUC of loaded model")
MODEL_RECALL = Gauge("churn_model_recall", "Hold-out recall of loaded model")
MODEL_TRAINED_TS = Gauge("churn_model_trained_timestamp_seconds", "Training time of loaded model")
MODEL_LOADED = Gauge("churn_model_loaded", "1 if a model is loaded")
