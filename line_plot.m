% --- Step 4: Compare Predicted vs. Actual Braking Force ---

clc; clear;

% Load Trained Model and Validation Data
load('trained_braking_model.mat', 'model');
load('validation_data.mat', 'X_val', 'Y_val');

% Convert table to array if necessary
if istable(X_val)
    X_val = table2array(X_val);
end

% Generate predictions
Y_pred = predict(model, X_val);

% Create enhanced comparison plot
figure;
plot(Y_val, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Actual'); 
hold on;
plot(Y_pred, 'r--', 'LineWidth', 2.2, 'DisplayName', 'Predicted');

% Add plot customization
xlabel('Sample Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Braking Force (N)', 'FontSize', 12, 'FontWeight', 'bold');
title('ANN Performance: Actual vs Predicted Braking Force', ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'LineWidth', 1.5, 'FontSize', 10);

% Calculate and display metrics with precision
mse = mean((Y_val - Y_pred).^2);  % Fixed missing parenthesis
mae = mean(abs(Y_val - Y_pred));
fprintf('=== Model Performance Metrics ===\n');
fprintf('Mean Squared Error (MSE): \t%.4f N²\n', mse);
fprintf('Mean Absolute Error (MAE): \t%.4f N\n\n', mae);

% Add correlation analysis
[R, P] = corrcoef(Y_val, Y_pred);
fprintf('Correlation Coefficient: \t%.4f\n', R(2,1));
fprintf('P-value: \t\t\t%.4e\n', P(2,1));

disp('Braking force comparison completed successfully!');