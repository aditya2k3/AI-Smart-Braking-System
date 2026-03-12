% --- Step 4: Compare Predicted vs. Actual Braking Force ---
clc; clear;

% Load Trained Model
load('trained_braking_model.mat', 'model');

% Load Validation Data
load('validation_data.mat', 'X_val', 'Y_val');

% Ensure X_val is a numeric array
if istable(X_val)
    X_val = table2array(X_val); % Convert table to numeric array if necessary
end

% Predict Braking Force for Validation Set
Y_pred = predict(model, X_val);

% Calculate Differences and Means
differences = Y_val - Y_pred;
means = (Y_val + Y_pred) / 2;

% Plot Bland-Altman Plot
figure;
scatter(means, differences, 'filled');
xlabel('Mean of Actual and Predicted Braking Force');
ylabel('Difference (Actual - Predicted)');
title('Bland-Altman Plot');
ylim([-max(abs(differences)) max(abs(differences))]); % Symmetric y-axis
hold on;
plot([min(means), max(means)], [0, 0], 'r--'); % Zero difference line
grid on;

% Calculate Performance Metrics
mse = mean((Y_val - Y_pred).^2); % Mean Squared Error
mae = mean(abs(Y_val - Y_pred)); % Mean Absolute Error
fprintf('Mean Squared Error (MSE): %.4f\n', mse);
fprintf('Mean Absolute Error (MAE): %.4f\n', mae);

disp('Comparison of predicted vs. actual braking force completed!');