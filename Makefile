.PHONY: help install lint typecheck test pipeline train-quick api up down bootstrap traffic traffic-drift drift fmt k8s-validate
PY ?= python
export PYTHONPATH := src

help:            ## show targets
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'
install:         ## install dev dependencies
	pip install -r requirements-dev.txt
lint:            ## ruff
	ruff check src tests airflow scripts
fmt:             ## auto-fix lint
	ruff check --fix src tests airflow scripts
typecheck:       ## mypy
	mypy src
test:            ## unit+integration+data+model tests with coverage gate (>=80%)
	pytest --cov --cov-report=term-missing --cov-report=xml
pipeline:        ## run the whole ML pipeline locally (no Airflow)
	$(PY) -m churn.cli run-all
train-quick:     ## fast smoke training (CI)
	$(PY) -m churn.cli run-all --quick
api:             ## run API locally on :8000
	uvicorn api.main:app --reload --port 8000
up:              ## start the full stack (API, Airflow, MLflow, MinIO, Prometheus, Grafana, Alertmanager)
	docker compose up -d --build
down:            ## stop the stack
	docker compose down
sync-storage:    ## sync data, reports, and artifacts to MinIO
	$(PY) -m churn.cli sync-all
bootstrap:       ## first run: train a champion locally so the API is ready, then restart the API
	$(PY) -m churn.cli run-all && docker compose restart api
traffic:         ## healthy traffic
	$(PY) scripts/simulate_traffic.py --n 300
traffic-drift:   ## drifted traffic -> triggers Evidently + Telegram alert
	$(PY) scripts/simulate_traffic.py --n 300 --drift
drift:           ## run the Evidently check manually
	$(PY) -m churn.cli drift
k8s-validate:    ## render manifests
	kubectl kustomize k8s/overlays/dev >/dev/null && kubectl kustomize k8s/overlays/prod >/dev/null && echo OK
k8s-up:          ## start kind cluster + install ArgoCD & metrics-server
	kind create cluster --config k8s/kind-config.yaml || true
	kubectl apply --server-side --force-conflicts -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
	kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
	kubectl patch deployment metrics-server -n kube-system --type 'json' -p '[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]' || true
	kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "NodePort", "ports": [{"name": "https", "port": 443, "targetPort": 8080, "nodePort": 30443}]}}' || true
k8s-down:        ## delete kind cluster
	kind delete cluster --name churn-cluster
argocd-apps:     ## apply ArgoCD Project and Applications
	kubectl apply -f argocd/project.yaml
	kubectl apply -f argocd/application-dev.yaml
	kubectl apply -f argocd/application-prod.yaml
