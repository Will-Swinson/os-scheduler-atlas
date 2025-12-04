from data_generator import WorkloadGenerator
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from sklearn.metrics import accuracy_score

generator = WorkloadGenerator()
training_data = generator.generate_training_dataset(1000)


feature_engineer = FeatureEngineer()
training_features = feature_engineer.prepare_ml_dataset(training_data)

print(training_features.head())


model = ModelTrainer()
dataset = model.create_target_labels(training_features)

(X, y) = model.prepare_training_dataset(dataset)

training_results = model.train_model(X, y)


print(training_results)

test_data = generator.generate_training_dataset(100)
test_features = feature_engineer.prepare_ml_dataset(test_data)
test_labeled = model.create_target_labels(test_features)

predictions = model.predict_best_algorithm(test_labeled)
actual_best = test_labeled["best_algorithms"].values


real_accuray = accuracy_score(actual_best, predictions)

print(real_accuray)
