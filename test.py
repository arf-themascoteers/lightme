from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="conv15", folds=10, algorithms=[
        "ann_conv"
    ])
    c.process()