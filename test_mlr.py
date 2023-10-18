from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="mlr", folds=10, algorithms=["mlr"])
    c.process()