N_DATASETS = 15;
METODO = "trainlm";
hiddenLayerSize = [8 4];
division = [70 15 15];
dataset = [];
for i = 1:N_DATASETS
    datos = csvread("datos_entrenamiento/ej"+i+".txt");
    filas_con_cero = any(datos(:,1:7) == 0, 2);
    datos(filas_con_cero, :) = [];
    dataset = [dataset ; datos];
end

numfilas = size(dataset, 1);
indicepermutacion = randperm(numfilas);
dataset = dataset(indicepermutacion, :);
inputs = dataset(:,1:7);
targets = dataset(:,8:9);

inputs = inputs';
targets = targets';

net = feedforwardnet(hiddenLayerSize, METODO);

net.divideParam.trainRatio = division(1)/100;
net.divideParam.valRatio = division(2)/100;
net.divideParam.testRatio = division(3)/100;

[net,tr] = train(net,inputs,targets);

outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);

%% 

gensim(net);