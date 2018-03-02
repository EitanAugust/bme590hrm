files = dir('*.csv');
for file = files'
    array = csvread(file.name)
    figure(1); clf
    plot(array(:,1), array(:,2), 'k-')
    title(file.name)
    pause(2)
end