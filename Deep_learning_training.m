clc; clear;

% --- Load Dataset ---
data = readtable('braking_data.csv');

% --- Prepare Data for Training ---
X = table2array(data(:, 1:5));           % Input features: Speed, Distance, Weather, IR, UV
Y = data.braking_force;                   % Output: Braking Force

% --- Split into Training and Validation ---
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_val = X(test(cv), :);
Y_val = Y(test(cv));

% --- Create Deep Learning Model ---
layers = [
    featureInputLayer(5)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {X_val, Y_val}, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% --- Train the Model ---
model = trainNetwork(X_train, Y_train, layers, options);

% --- Save the Trained Model ---
save('braking_model.mat', 'model');
disp('Model trained and saved successfully!');
