%  Step 1: Generate Artificial Data ---
clc; clear;

% Parameters
num_samples = 1000; % Number of data points

% Input Features
speed = randi([10, 120], num_samples, 1); % Vehicle speed in km/h
distance_to_obstacle = randi([1, 50], num_samples, 1); % Distance to obstacle in meters
weather_condition = randi([1, 3], num_samples, 1); % 1: Clear, 2: Rainy, 3: Snowy
ir_sensor_reading = randi([50, 255], num_samples, 1); % IR sensor reading (normalized)
uv_sensor_reading = randi([0, 100], num_samples, 1); % UV sensor reading (normalized)

% Output: Braking Force (calculated based on a simple rule-based model)
braking_force = zeros(num_samples, 1);
for i = 1:num_samples
    if weather_condition(i) == 1 % Clear weather
        braking_force(i) = max(0, 0.5 * speed(i) - 0.2 * distance_to_obstacle(i) + 0.1 * ir_sensor_reading(i));
    elseif weather_condition(i) == 2 % Rainy weather
        braking_force(i) = max(0, 0.7 * speed(i) - 0.3 * distance_to_obstacle(i) + 0.2 * ir_sensor_reading(i));
    else % Snowy weather
        braking_force(i) = max(0, 0.9 * speed(i) - 0.4 * distance_to_obstacle(i) + 0.3 * ir_sensor_reading(i));
    end
end

% Combine into a table
data = table(speed, distance_to_obstacle, weather_condition, ir_sensor_reading, uv_sensor_reading, braking_force);

% Save the artificial dataset
writetable(data, 'artificial_braking_data.csv');
disp('Artificial dataset generated and saved successfully!');