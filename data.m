%% Smart Braking System: ANN vs PID Analysis
% Step 1: Synthetic Data Generation (Replace with real sensor data later)
clc;
clear;
rng(42); % For reproducibility

% Simulation parameters
num_samples = 5000;
time = linspace(0, 10, num_samples);

% Synthetic sensor data generation
[vehicle_speed, obstacle_distance, road_friction] = generate_sensor_data(num_samples);

% Physics-based braking distance calculation (Ground truth)
deceleration = 0.8 * road_friction * 9.81; % m/s²
true_braking_distance = (vehicle_speed.^2) ./ (2 * deceleration);

% Add sensor noise characteristics
noise_level = 0.05;
sensor_distance = obstacle_distance .* (1 + noise_level * randn(size(obstacle_distance)));

%% Step 2: ANN Training (Braking Distance Prediction)
% Normalize data
input_data = [vehicle_speed; sensor_distance; road_friction]';
output_data = true_braking_distance';

% Create and configure ANN
net = feedforwardnet([10 8]); % Two hidden layers: 10 and 8 neurons
net.trainFcn = 'trainlm'; % Levenberg-Marquardt
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train ANN
[net, tr] = train(net, input_data', output_data');

% ANN prediction
ann_predicted_distance = net(input_data')';

%% Step 3: PID Controller Implementation (Comparative Baseline)
Kp = 0.85; Ki = 0.01; Kd = 0.2;
pid_braking_force = zeros(size(time));
integral = 0; prev_error = 0;

for i = 1:num_samples
    error = sensor_distance(i) - ann_predicted_distance(i);
    integral = integral + error;
    derivative = error - prev_error;
    
    % Anti-windup protection
    if abs(integral) > 100
        integral = sign(integral) * 100;
    end
    
    pid_braking_force(i) = Kp*error + Ki*integral + Kd*derivative;
    prev_error = error;
end

%% Step 4: Visualization and Analysis
figure;

% Subplot 1: Braking Distance Comparison
subplot(3,1,1);
plot(time, true_braking_distance, 'b', 'LineWidth', 1.5);
hold on;
plot(time, ann_predicted_distance, '--r', 'LineWidth', 1.5);
title('Braking Distance: Actual vs ANN Prediction');
xlabel('Time (s)');
ylabel('Distance (m)');
legend('Actual', 'ANN Prediction');
grid on;

% Calculate RMSE
rmse = sqrt(mean((true_braking_distance - ann_predicted_distance).^2));
annotation('textbox', [0.15, 0.75, 0.1, 0.1], 'String', ...
    sprintf('RMSE: %.2f m', rmse), 'EdgeColor', 'none');

% Subplot 2: Sensor Efficiency Analysis
subplot(3,1,2);
sensor_error = abs(sensor_distance - obstacle_distance)./obstacle_distance * 100;
window_size = 50;
moving_avg = movmean(sensor_error, window_size);
plot(time, sensor_error, 'Color', [0.5 0.5 0.5]);
hold on;
plot(time, moving_avg, 'm', 'LineWidth', 2);
title('Sensor Detection Efficiency');
xlabel('Time (s)');
ylabel('Error (%)');
legend('Instant Error', ['Moving Avg (' num2str(window_size) '-sample)']);
grid on;

% Subplot 3: Braking Force Comparison
subplot(3,1,3);
ann_braking_force = ann_predicted_distance ./ (vehicle_speed + eps); % Simplified force estimation
plot(time, ann_braking_force, 'g', 'LineWidth', 1.5);
hold on;
plot(time, pid_braking_force, '--k', 'LineWidth', 1.5);
title('Braking Force: ANN vs PID Controller');
xlabel('Time (s)');
ylabel('Normalized Force');
legend('ANN Force', 'PID Force');
grid on;

%% Helper Function: Sensor Data Generation
function [speed, distance, friction] = generate_sensor_data(n)
    % Vehicle speed (m/s) with realistic acceleration profile
    speed = 15 + 3*tanh(linspace(-3, 3, n)) + 0.5*randn(1, n);
    
    % Obstacle distance (m) with emergency scenarios
    distance = 50 - cumsum(0.1 + 0.05*randn(1, n));
    emergency_points = randperm(n, 5);
    distance(emergency_points) = distance(emergency_points) - 15;
    
    % Road friction coefficient (0.1-0.9)
    friction = 0.3 + 0.5*sigmoid(linspace(-5, 5, n)) + 0.1*randn(1, n);
    friction = max(min(friction, 0.9), 0.1);
end

function y = sigmoid(x)
    y = 1./(1 + exp(-x));
end