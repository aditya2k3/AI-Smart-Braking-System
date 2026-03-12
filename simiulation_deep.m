clc; clear;

% Load Model and Data 
load('braking_model.mat');
data = readtable('braking_data.csv');

% Simulate Real-World Scenarios 
num_tests = 50;
test_data = table2array(data(1:num_tests, 1:5));
actual_force = data.braking_force(1:num_tests); % Extract actual values

% Predict Braking Force using DL Model
predicted_force = predict(model, test_data);

% Calculate Performance Metrics
mse = mean((actual_force - predicted_force).^2);  % Mean Squared Error
rmse = sqrt(mse);                                 % Root Mean Squared Error
mae = mean(abs(actual_force - predicted_force));  % Mean Absolute Error

% Calculate R-squared (Coefficient of Determination)
ss_res = sum((actual_force - predicted_force).^2);
ss_tot = sum((actual_force - mean(actual_force)).^2);
r2 = 1 - (ss_res/ss_tot);

% Display Metrics
fprintf('\nBraking Force Prediction Performance:\n');
fprintf('======================================\n');
fprintf('Mean Squared Error (MSE): \t%.4f N²\n', mse);
fprintf('Root Mean Squared Error (RMSE): \t%.4f N\n', rmse);
fprintf('Mean Absolute Error (MAE): \t%.4f N\n', mae);
fprintf('R-squared (Efficiency): \t%.4f\n', r2);
fprintf('======================================\n');

% Plot Results 
figure;
plot(1:num_tests, actual_force, 'b-o', 'LineWidth', 2);
hold on;
plot(1:num_tests, predicted_force, 'r-*', 'LineWidth', 2);
xlabel('Test Cases');
ylabel('Braking Force (N)');
legend('Actual Force', 'Predicted Force', 'Location', 'best');
title('Deep Learning Braking Force Prediction');
grid on;

disp('Simulation with DL Model completed!');;
