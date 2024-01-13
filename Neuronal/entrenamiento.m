% Carga de datos de ejemplo disponibles en la toolbox

% Código equivalente al que da la práctica para simplefit_dataset
% [inputs,targets] = simplefit_dataset;
% METODOS = ["trainlm"];
% DIVISIONES = [70 15 15];
% PAUSAR = true;

% Código para bodyfat_dataset
N_DATASETS = 15;
dataset = []
for i = 1:N_DATASETS
    datos = csvread("datos_neuronal/ej"+i+".txt");
    % Elimina las filas con al menos un 0 en las primeras 5 columnas
    filas_con_cero = any(datos(:,1:7) == 0, 2);
    datos(filas_con_cero, :) = [];
    dataset = [dataset ; datos];
end

mdl = fitlm(dataset(:, 1), dataset(:, 2));

distances = mdl.predict - dataset(:, 2);

threshold = 2.25 * mean(abs(distances));

outliers = abs(distances) > threshold;

dataset = dataset(~outliers, :);

numfilas = size(dataset, 1);
indicepermutacion = randperm(numfilas);
dataset = dataset(indicepermutacion, :);
inputs = dataset(:,1:7);
targets = dataset(:,8:9);

inputs = inputs';
targets = targets';

disp(inputs)

METODO = "trainlm";

% Creación de la red
hiddenLayerSize = [8 4];

net = feedforwardnet(hiddenLayerSize, METODO);

division = [70 15 15];
net.divideParam.trainRatio = division(1)/100;
net.divideParam.valRatio = division(2)/100;
net.divideParam.testRatio = division(3)/100;

[net,tr] = train(net,inputs,targets);

outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);

%% 

gensim(net);

save("netData.mat", "net");




