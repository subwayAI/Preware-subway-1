function varargout = untitled(varargin)
% UNTITLED MATLAB code for untitled.fig
%      UNTITLED, by itself, creates a new UNTITLED or raises the existing
%      singleton*.
%
%      H = UNTITLED returns the handle to a new UNTITLED or the handle to
%      the existing singleton*.
%
%      UNTITLED('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UNTITLED.M with the given input arguments.
%
%      UNTITLED('Property','Value',...) creates a new UNTITLED or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before untitled_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to untitled_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help untitled

% Last Modified by GUIDE v2.5 28-Aug-2023 17:56:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @untitled_OpeningFcn, ...
                   'gui_OutputFcn',  @untitled_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before untitled is made visible.
function untitled_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to untitled (see VARARGIN)

% Choose default command line output for untitled
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes untitled wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = untitled_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)








NOMAL=[]
d='./data/normal/normal ('
for dd=1:52
cc=strcat(d,num2str(dd),')')

tezhneg=[]
for uu=1:15
%load 3.mat

sx={'Peak X-axis acceleration',...
    'X-axis acceleration valid value',...
    'X axis speed value',...
    'X axis displacement',...	
    'X-axis natural frequency',...
    'Peak y-axis acceleration',...	
    'y-axis acceleration valid value',...	
    'y axis speed value',...
    'y axis displacement',...	
    'y-axis natural frequency',...
    'Peak z-axis acceleration',...	
    'z-axis acceleration valid value',...	
    'z axis speed value',...
    'Z axis displacement',...	
    'z-axis natural frequency'
    };
cc,sx{uu}
num = xlsread(cc,sx{uu});
D=tezheng(num);
tezhneg=[tezhneg,D];
end
NOMAL=[NOMAL;tezhneg];

end


NOMAL=[]
d='./data/slightvibration/slightvibration ('
for dd=1:108
cc=strcat(d,num2str(dd),')')

tezhneg=[]
for uu=1:15
%load 3.mat

sx={'Peak X-axis acceleration',...
    'X-axis acceleration valid value',...
    'X axis speed value',...
    'X axis displacement',...	
    'X-axis natural frequency',...
    'Peak y-axis acceleration',...	
    'y-axis acceleration valid value',...	
    'y axis speed value',...
    'y axis displacement',...	
    'y-axis natural frequency',...
    'Peak z-axis acceleration',...	
    'z-axis acceleration valid value',...	
    'z axis speed value',...
    'Z axis displacement',...	
    'z-axis natural frequency'
    };
num = xlsread(cc,sx{uu});
D=tezheng(num);
tezhneg=[tezhneg,D];
end
NOMAL=[NOMAL;tezhneg];

end
save sli.mat NOMAL






NOMAL=[]
d='./data/judder/judder ('
for dd=1:183
cc=strcat(d,num2str(dd),')')

tezhneg=[]
for uu=1:15
%load 3.mat

sx={'Peak X-axis acceleration',...
    'X-axis acceleration valid value',...
    'X axis speed value',...
    'X axis displacement',...	
    'X-axis natural frequency',...
    'Peak y-axis acceleration',...	
    'y-axis acceleration valid value',...	
    'y axis speed value',...
    'y axis displacement',...	
    'y-axis natural frequency',...
    'Peak z-axis acceleration',...	
    'z-axis acceleration valid value',...	
    'z axis speed value',...
    'Z axis displacement',...	
    'z-axis natural frequency'
    };
num = xlsread(cc,sx{uu});
D=tezheng(num);
tezhneg=[tezhneg,D];
end
NOMAL=[NOMAL;tezhneg];

end
save judder.mat NOMAL

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
x=[]
lei=[]
load('nomal.mat')
x=[x;NOMAL];
lei=[lei;ones(size(NOMAL,1),1)]
load('sli.mat')
x=[x;NOMAL];
lei=[lei;2*ones(size(NOMAL,1),1)]
load('judder.mat')
x=[x;NOMAL];
lei=[lei;3*ones(size(NOMAL,1),1)]
figure
plot(x')
plot(lei)

save 33.mat x  lei

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


%% 数据提取

% 载入测试数据subway_eq,其中包含的数据为classnumber = 3,subway_eq:178*13的矩阵,subway_eq_labes:178*1的列向量
load ('a/2.mat');



format compact;
%% 数据提取

% 载入测试数据subway_eq,其中包含的数据为classnumber = 3,subway_eq:178*13的矩阵,subway_eq_labes:178*1的列向量

%data=[ones(120,1);2*ones(120,1);3*ones(120,1)];
%s1=[cc,data];
data=x%(:,1:end-1)
subway_eq_label=lei
%[ones(120,1);2*ones(120,1);3*ones(120,1)];
data=[data,subway_eq_label]
%从1到2000间随机排序
k=rand(1,343);
[m,n]=sort(k);

%输入输出数据


%随机提取1500个样本为训练样本，500个样本为预测样本
inx=data(n(1:343),:);
%output_train=output(n(1:360),:);





subway_eq=inx(:,1:end-1)
subway_eq_label=inx(:,end)
% 画出测试数据的box可视化图
;






% 选定训练集和测试集
%subway_eq=[subway_eq,subway_eq,subway_eq,subway_eq,subway_eq];
ccc=300
% 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
train_subway_eq = [subway_eq(1:ccc,:)]%出来
train_subway_eq_label = [subway_eq_label(1:ccc)];%;subway_eq_label(60:95);subway_eq_label(131:153)];
% 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
test_subway_eq = [subway_eq(ccc+1:end,:)]%;subway_eq(96:130,:);subway_eq(154:178,:)];
% 相应的测试集的标签也要分离出来
test_subway_eq_label = [subway_eq_label(ccc+1:end)]%;subway_eq_label(96:130);subway_eq_label(154:178)];

%% 数据预处理
[mtrain,ntrain] = size(train_subway_eq);
[mtest,ntest] = size(test_subway_eq);

dataset = [train_subway_eq;test_subway_eq];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_subway_eq = dataset_scale(1:mtrain,:);
test_subway_eq = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% SVM网络训练


p_train=train_subway_eq'
p_test=test_subway_eq'
%% 将期望类别转换为向量
t_train=train_subway_eq_label 
t_test=test_subway_eq_label
t_train=ind2vec(t_train');
t_train_temp=train_subway_eq_label %Train(:,4)';
%% 使用newpnn函数建立PNN SPREAD选取为1.5
Spread=10.005;
net=newpnn(p_train,t_train,Spread)

%% 训练数据回代 查看网络的分类效果

% Sim函数进行网络预测
Y=sim(net,p_train);
% 将网络输出向量转换为指针
Yc=vec2ind(Y);

%% 通过作图 观察网络对训练数据分类效果
figure

stem(1:length(Yc),Yc,'bo')
hold on
stem(1:length(Yc),t_train_temp,'r*')
xlabel('sample')
ylabel('result')

H=Yc-t_train_temp';

s=sum(H==0)
s1s=s/length(Yc)

title(['train data PNN ,acc:',num2str(s1s)])



%% 网络预测未知数据效果
Y2=sim(net,p_test);
Y2c=vec2ind(Y2);
figure(12)
stem(1:length(Y2c),Y2c,'b^')
hold on
stem(1:length(Y2c),t_test,'r*')
stem(1:length(Y2c),t_test,'r*')
title('test data PNN result ')
xlabel('sample number')
ylabel('result')
set(gca,'Ytick',[1:5])
s=sum(Y2c==t_test')
ss=s/length(Y2c)
title(['test data PNN ,acc:',num2str(ss)])

figure %创建混淆矩阵图
cm = confusionchart(t_test',Y2c)

cm.Title = 'pnn test data test data Confusion Matrix';


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load ('a/2.mat');
%data=[ones(120,1);2*ones(120,1);3*ones(120,1)];
%s1=[cc,data];
data=x;%(:,1:end-1)
subway_eq_label=lei;%[ones(120,1);2*ones(120,1);3*ones(120,1)];
data=[data,subway_eq_label]
%从1到2000间随机排序
k=rand(1,343);
[m,n]=sort(k);

%输入输出数据


%随机提取1500个样本为训练样本，500个样本为预测样本
inx=data(n(1:343),:);
%output_train=output(n(1:360),:);





subway_eq=inx(:,1:end-1)
subway_eq_label=inx(:,end)


% 选定训练集和测试集
%subway_eq=[subway_eq,subway_eq,subway_eq,subway_eq,subway_eq];
ccc=300
% 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
train_subway_eq = [subway_eq(1:ccc,:)]%出来
train_subway_eq_label = [subway_eq_label(1:ccc)];%;subway_eq_label(60:95);subway_eq_label(131:153)];
% 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
test_subway_eq = [subway_eq(ccc+1:end,:)]%;subway_eq(96:130,:);subway_eq(154:178,:)];
% 相应的测试集的标签也要分离出来
test_subway_eq_label = [subway_eq_label(ccc+1:end)]%;subway_eq_label(96:130);subway_eq_label(154:178)];

%% 数据预处理
% 数据预处理,将训练集和测试集归一化到[0,1]区间

[mtrain,ntrain] = size(train_subway_eq);
[mtest,ntest] = size(test_subway_eq);

dataset = [train_subway_eq;test_subway_eq];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_subway_eq = dataset_scale(1:mtrain,:);
test_subway_eq = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% SVM网络训练
model = svmtrain(train_subway_eq_label, train_subway_eq, '-c 10.02 -g 10.1');

%% SVM网络预测
[predict_label, accuracy] = svmpredict(test_subway_eq_label, test_subway_eq, model);

%% 结果分析

% 测试集的实际分类和预测分类图
% 通过图可以看出只有一个测试样本是被错分的
figure;
hold on;
plot(test_subway_eq_label,'o');
plot(predict_label,'r*');
xlabel('SAMPLE','FontSize',12);
ylabel('label','FontSize',12);
legend('real','predict');
title('classify','FontSize',12);
grid on;
cc=sum(test_subway_eq_label-predict_label==0)/length(test_subway_eq_label)
title(['train data svm ,acc:',num2str(cc)])

figure %创建混淆矩阵图
cm = confusionchart(test_subway_eq_label,predict_label)

cm.Title = 'svm Confusion Matrix';
%cm.Xlable = 'svm Confusion Matrix';



% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 载入测试数据subway_eq,其中包含的数据为classnumber = 3,subway_eq:178*13的矩阵,subway_eq_labes:178*1的列向量
load ('a/2.mat');
%data=[ones(120,1);2*ones(120,1);3*ones(120,1)];
%s1=[cc,data];
data=x%(:,1:end-1)
subway_eq_label=lei;%[ones(120,1);2*ones(120,1);3*ones(120,1)];
data=[data,subway_eq_label]
%从1到2000间随机排序
k=rand(1,343);
[m,n]=sort(k);

%输入输出数据


%随机提取1500个样本为训练样本，500个样本为预测样本
inx=data(n(1:343),:);
%output_train=output(n(1:360),:);





subway_eq=inx(:,1:end-1)
subway_eq_label=inx(:,end)


% 选定训练集和测试集
%subway_eq=[subway_eq,subway_eq,subway_eq,subway_eq,subway_eq];
ccc=320
% 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
train_subway_eq = [subway_eq(1:ccc,:)]%出来
train_subway_eq_label = [subway_eq_label(1:ccc)];%;subway_eq_label(60:95);subway_eq_label(131:153)];
% 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
test_subway_eq = [subway_eq(ccc+1:end,:)]%;subway_eq(96:130,:);subway_eq(154:178,:)];
% 相应的测试集的标签也要分离出来
test_subway_eq_label = [subway_eq_label(ccc+1:end)]%;subway_eq_label(96:130);subway_eq_label(154:178)];

%% 数据预处理


train_x= train_subway_eq';
YTrain = train_subway_eq_label';

test_x=test_subway_eq';
test_y=test_subway_eq_label';


si=length(train_subway_eq)

train_y=YTrain
method=@mapminmax;
% method=@mapstd;
[train_x,train_ps]=method(train_x);
test_x=method('apply',test_x,train_ps);
[train_y,output_ps]=method(train_y);
test_y=method('apply',test_y,output_ps);



fs=size(train_x )
fs1=size(test_x )
trainD=reshape(train_x,[fs(1),1,1,fs(2)]);%训练集输入
testD=reshape(test_x,[fs1(1),1,1,fs1(2)]);%测试集输入
targetD = train_y;%训练集输出
targetD_test  = test_y;%测试集输出



 layers = [
    imageInputLayer([fs(1) 1 1]) %输入层参数设置
    convolution2dLayer([3,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
    convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    reluLayer%relu激活函数
   %convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
  %  reluLayer%relu激活函数convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
   % reluLayer%relu激活函数 %convolution2dLayer([1,1],32,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    %reluLayer%relu激活函数
    %convolution2dLayer([1,1],16,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    %reluLayer%relu激活函数
    %convolution2dLayer([1,1],8,'Padding','same')%卷积层的核大小[3 1],因为我们的输入是[209 1],是一维的数据，所以卷积核第二个参数为1就行了，这样就是1d卷积
%、数量，填充方式
    %reluLayer%relu激活函数
    maxPooling2dLayer([1 1],'Stride',2)% 2x1 kernel stride=2
    fullyConnectedLayer(384) % 384 全连接层神经元
    %fullyConnectedLayer(256) % 384 全连接层神经元
    %reluLayer%relu激活函数
    fullyConnectedLayer(128) % 384 全连接层神经元
    reluLayer%relu激活函数
    fullyConnectedLayer(64) % 384 全连接层神经元
    reluLayer%relu激活函数
    fullyConnectedLayer(32) % 384 全连接层神经元
    reluLayer%relu激活函数
    fullyConnectedLayer(1) % 输出层神经元
    regressionLayer];%添加回归层，用于计算损失值
 FiltZise=[2,1]

%创建"CNN-LSTM"模型
    layers1 = [...
        % 输入特征
        imageInputLayer([fs(1) 1 1],'Name','input')
        sequenceFoldingLayer('Name','fold')
        % CNN特征提取
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        averagePooling2dLayer(1,'Stride',FiltZise,'Name','pool1')
        % 展开层
        sequenceUnfoldingLayer('Name','unfold')
        % 平滑层
        flattenLayer('Name','flatten')
        % LSTM特征学习
        lstmLayer(128,'Name','lstm1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.5,'Name','drop1')
        lstmLayer(64,'Name','lstm2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.5,'Name','drop2')
        lstmLayer(32,'Name','lstm3','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.5,'Name','drop3')
        % LSTM输出
        lstmLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.5,'Name','drop4')
        lstmLayer(16,'Name','lstm4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.5,'Name','drop5')
        % 全连接层
        fullyConnectedLayer(1,'Name','fc')
        regressionLayer('Name','output')    ];

    layers1 = layerGraph(layers1);
    layers1 = connectLayers(layers1,'fold/miniBatchSize','unfold/miniBatchSize');


% 设置迭代次数 batchsize 学习率啥的
options = trainingOptions('adam', ...
    'MaxEpochs',112, ...
    'MiniBatchSize',256, ...
    'InitialLearnRate',0.001, ...
    'GradientThreshold',1, ...
    'Verbose',false,...
    'Plots','training-progress')%,...
    %'ValidationData',{testD,targetD_test'}
%);
%这里要吐槽一下，输入数据都是最后一维为样本数，偏偏输出要第一维为样本数，所以targetD和targetD_test都取了转置
% 训练

%  创建"CNN-LSTM"模型
FiltZise=2
    layers2 = [...
        % 输入特征
        sequenceInputLayer([fs(1) 1 1],'Name','input')
        sequenceFoldingLayer('Name','fold')
        % CNN特征提取
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        averagePooling2dLayer(1,'Stride',FiltZise,'Name','pool1')
        % 展开层
        sequenceUnfoldingLayer('Name','unfold')
        % 平滑层
        flattenLayer('Name','flatten')
        % LSTM特征学习
        lstmLayer(128,'Name','lstm1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop1')
        % LSTM输出
        lstmLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
         lstmLayer(32,'OutputMode',"last",'Name','bil5','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
       
        dropoutLayer(0.25,'Name','drop2')
        % 全连接层
        fullyConnectedLayer(5,'Name','fc')
        %fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer    ];

%    layers2 = layerGraph(layers2);
    %layers2 = connectLayers(layers2,'fold/miniBatchSize','unfold/miniBatchSize');

    
net = trainNetwork(trainD,targetD',layers,options);
% 预测
YPred = predict(net,testD);
%YPred =  classify(net,testD);
% 结果
YPred=double(YPred');%输出是n*1的single型数据，要转换为1*n的double是数据形式
% 反归一化
predict_value=method('reverse',YPred,output_ps);predict_value=double(predict_value);
true_value=method('reverse',targetD_test,output_ps);true_value=double(true_value);


 

 
 ss=sum(true_value'-predict_value==0)/length(predict_value)
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['(RMSE)：',num2str(rmse)])
mae=mean(abs(true_value-predict_value));
disp(['（MAE）：',num2str(mae)])
mape=mean(abs((true_value-predict_value)./true_value));
disp(['（MAPE）：',num2str(mape*100),'%'])



predict_value=round(predict_value)';
figure
plot(true_value,'-','linewidth',1)
hold on
plot(predict_value,'-','linewidth',1)
legend('test data cnn real','smooth cnn-lstm predict')
grid on
 
 
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['(RMSE)：',num2str(rmse)])
mae=mean(abs(true_value-predict_value));
disp(['（MAE）：',num2str(mae)])
mape=mean(abs((true_value-predict_value)./true_value));
disp(['（MAPE）：',num2str(mape*100),'%'])

ss=sum(true_value'-predict_value==0)/length(predict_value)
figure %创建混淆矩阵图
cm = confusionchart(true_value,round(predict_value))

cm.Title = 'cnn  test data Confusion Matrix';



figure
YPred = predict(net,trainD);
%YPred =  classify(net,testD);
% 结果
YPred=double(YPred');%输出是n*1的single型数据，要转换为1*n的double是数据形式
% 反归一化
predict_value=method('reverse',YPred,output_ps);predict_value=double(predict_value);
true_value=method('reverse',targetD,output_ps);true_value=double(true_value);
plot(true_value,'-','linewidth',1)
hold on
plot(predict_value,'-','linewidth',1)
legend('trian data cnn real','cnn-lstm predict')
grid on
ss1=sum(true_value-round(predict_value)==0)/length(predict_value)
title(['train data cnn ,acc:',num2str(ss1)])

figure %创建混淆矩阵图
cm = confusionchart(true_value,round(predict_value))

cm.Title = 'cnn lstm Confusion Matrix';
%save  cnnlstm.mat 


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ddd1
