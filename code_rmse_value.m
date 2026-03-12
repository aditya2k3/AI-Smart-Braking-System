% --- Step 4: Calculate Performance Metrics ---
clc; clear;

% Load Trained Model
load('trained_braking_model.mat', 'model');

% Load Validation Data
load('validation_data.mat', 'X_val', 'Y_val');

% Ensure X_val is numeric
if istable(X_val)
    X_val = table2array(X_val);
end

% Get predictions
Y_pred = predict(model, X_val);

% Calculate core metrics
mse = mean((Y_val - Y_pred).^2);      % Mean Squared Error
rmse = sqrt(mse);                     % Root Mean Squared Error
mae = mean(abs(Y_val - Y_pred));      % Mean Absolute Error

% Calculate R-squared (Efficiency)
ss_res = sum((Y_val - Y_pred).^2);    % Residual sum of squares
ss_tot = sum((Y_val - mean(Y_val)).^2); % Total sum of squares
r2 = 1 - (ss_res / ss_tot);           % Coefficient of determination

% Display results in command window
fprintf('\nBraking Force Prediction Performance:\n');
fprintf('======================================\n');
fprintf('Mean Squared Error (MSE): \t%.4f N²\n', mse);
fprintf('Root Mean Squared Error (RMSE): \t%.4f N\n', rmse);
fprintf('Mean Absolute Error (MAE): \t%.4f N\n', mae);
fprintf('R-squared (Efficiency): \t%.4f\n', r2);
fprintf('======================================\n');

disp('Metric calculation completed!');