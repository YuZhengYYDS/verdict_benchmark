@echo off
echo VERDICT Benchmark Model Evaluation
echo ==================================

echo.
echo Installing required packages...
pip install -r requirements_eval.txt

echo.
echo Running basic evaluation...
python evaluate_models.py --config configs/mlp.yaml --output-dir evaluation_results

echo.
echo Running advanced evaluation with statistical analysis...
python advanced_evaluate.py --config configs/mlp.yaml --output-dir advanced_evaluation

echo.
echo Evaluation complete! Check the following directories:
echo - evaluation_results/ (basic evaluation)
echo - advanced_evaluation/ (advanced statistical analysis)

pause
