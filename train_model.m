% --- Step 2: Train the ANN Model ---
clc; clear;

% Load Dataset
data = readtable('artificial_braking_data.csv');

% Prepare Data for Training
X = table2array(data(:, 1:5)); % Input features: Speed, Distance, Weather, IR, UV
Y = data.braking_force;       % Output: Braking Force

% Split into Training and Validation Sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_val = X(test(cv), :);
Y_val = Y(test(cv));

% Save Validation Data
save('validation_data.mat', 'X_val', 'Y_val'); % Save X_val and Y_val
disp('Validation data saved successfully!');

% Define ANN Architecture
layers = [
    featureInputLayer(5)          % Input layer with 5 features
    fullyConnectedLayer(64)       % Hidden layer with 64 neurons
    reluLayer                     % Activation function
    fullyConnectedLayer(32)       % Hidden layer with 32 neurons
    reluLayer                     % Activation function
    fullyConnectedLayer(1)        % Output layer (regression task)
    regressionLayer               % Regression layer
];

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_val, Y_val}, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the Model
model = trainNetwork(X_train, Y_train, layers, options);

% Save the Trained Model
save('trained_braking_model.mat', 'model');
disp('Model trained and saved successfully!');
