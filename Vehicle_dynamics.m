% --- Step 3: Visualize Vehicle Dynamics ---
figure;

% Plot Speed vs. Braking Force using Line Plot
subplot(2, 1, 1);
plot(data.speed, data.braking_force, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0.85, 0.33, 0.1]);
xlabel('Speed (km/h)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Braking Force (N)', 'FontSize', 12, 'FontWeight', 'bold');
title('Speed vs. Braking Force', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Braking Force', 'Location', 'northwest');

% Plot Distance to Obstacle vs. Braking Force using Bar Plot
subplot(2, 1, 2);
bar(data.distance_to_obstacle, data.braking_force, 'FaceColor', [0.2, 0.7, 0.9], 'EdgeColor', [0, 0.45, 0.74], 'LineWidth', 1.5);
xlabel('Distance to Obstacle (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Braking Force (N)', 'FontSize', 12, 'FontWeight', 'bold');
title('Distance to Obstacle vs. Braking Force', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
legend('Braking Force', 'Location', 'northeast');

% Display message
disp('Vehicle dynamics visualized successfully!');

