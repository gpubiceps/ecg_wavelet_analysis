function varargout = monit(varargin)
% MONIT MATLAB code for monit.fig
%      MONIT, by itself, creates a new MONIT or raises the existing
%      singleton*.
%
%      H = MONIT returns the handle to a new MONIT or the handle to
%      the existing singleton*.
%
%      MONIT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MONIT.M with the given input arguments.
%
%      MONIT('Property','Value',...) creates a new MONIT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before monit_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to monit_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help monit

% Last Modified by GUIDE v2.5 02-Apr-2020 19:15:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @monit_OpeningFcn, ...
                   'gui_OutputFcn',  @monit_OutputFcn, ...
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


% --- Executes just before monit is made visible.
function monit_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to monit (see VARARGIN)

% Choose default command line output for monit
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes monit wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = monit_OutputFcn(~, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function uitable2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uitable2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


function togglebutton2_Callback(hObject, eventdata, handles)
global RR;
%�����������
maxRR = max(RR);
maxRRp = maxRR*1.2;

axes(handles.axes2);
cla('reset');
hold on
title('�����������');
xlabel('n');
ylabel('RR, �');

for i=1:length(RR)
    x(1)=i;     
    x(2)=i;     
    R(1)=0;     
    R(2)=RR(i);     
    plot(x,R);      
end
xlim([0 length(RR)])
ylim([0 maxRRp])


function togglebutton3_Callback(hObject, eventdata, handles)
global RR;
% ����� ����� RR-����������
NRR=length(RR);
axes(handles.axes2);
cla('reset');
hold on
title('�������������');
xlabel('RR_i, �');
ylabel('RR_i+1, �');

plot(RR(1:NRR-1),RR(2:NRR),'.'); % ���������� �������������


function togglebutton4_Callback(hObject, eventdata, handles)
global RR;
%�����������
maxRR = max(RR);
dH=0.05;    % ��� ����������� (50 ��) 
X=0:dH:maxRR; % ���������� �� ��� ������� (RR-��������, �) 
H=histc(RR,X); % ������ ����������� 
SH=sum(H);  % ����� ����������� (����� RR-����������) 
PH=H/SH*100; % ��������� ����������� � % 

axes(handles.axes2);
cla('reset');
hold on
title('�����������');
xlabel('��');
ylabel('%');
bar(X,PH,'histc') % ���������� �����������
maxh=max(PH)*1.2;
ylim([0 maxh])



function edit1_Callback(hObject, eventdata, handles)

function edit1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function pushbutton1_Callback(hObject, eventdata, handles)

global Sig len_signal file all_peaks all_bounds RR;
file = uigetfile('*.mat');
Signal = load(file);
ECG = Signal.data(:,1);
ECG = detrend(ECG);
% ������� ���������� (��������� ������� ����������)
ecgden = wdenoise(ECG,6,'Wavelet','sym6','DenoisingMethod',...
'Bayes','ThresholdRule', 'Soft','NoiseEstimate','LevelDependent');

Fs = get(handles.edit1,'string');
Fs = str2double(Fs); % ������� �������������
slider_val = get(handles.slider2,'Value'); % �������� ��������

len_window = get(handles.edit2,'string'); % �������� ����� ����
len_window = str2double(len_window);
len_signal = length(ecgden);

i_start = round((len_signal - len_window*Fs)*slider_val)+1;
i_stop = i_start + len_window*Fs;

T = 1/Fs;
tmax = i_stop/Fs;
t = i_start/Fs:T:tmax;

%��� ����������
n=3; %������� �������
fc1=0.5; %������� �����
fc2=25; 
Wn=[fc1 fc2]*2/Fs; %������������ ������� �����
[b, a]= butter(n, Wn); %������ ������������� ������� �����������
Sig=filtfilt(b,a,ecgden); %���������� � ��� �������

%����������� ���������� ��������� QRS
n=3; %������� �������
fc1=5; %������� �����
fc2=15; 
Wn=[fc1 fc2]*2/Fs; %������������ ������� �����
[b, a]= butter(n, Wn); %������ ������������� ������� �����������
f_qrs = filtfilt(b,a,ecgden); %���������� � ��� �������
f_qrs = f_qrs/max(abs(f_qrs)); %������������

%������ �����������
b = [1 2 0 -2 -1].*(1/8)*Fs;
f_qrs_d = filtfilt(b,1,f_qrs);
f_qrs_d = f_qrs_d/max(abs(f_qrs_d));

%���������������
f_qrs_s = f_qrs_d.^2;

%���������� �������
f_qrs_m = conv(f_qrs_s ,ones(1 ,round(0.150*Fs))/round(0.150*Fs));

%������������
f_qrs_m = f_qrs_m/max(abs(f_qrs_m));


%����� ����� ����������� �������
[pks,locs] = findpeaks(f_qrs_m,'MinPeakDistance',round(0.2*Fs),'MinPeakHeight',0.1);

%������� ������� ������� ����� ������ ���� � �����
dist_n = 0.5*Fs;%����������� ��������� ����� �����������
locs_f = [];
pks_f = [];
locs_f(1) = locs(1);
pks_f(1) = pks(1);

for i=1:(length(locs)-1)
    if (locs(i+1)-locs(i)) > dist_n
        locs_f(end+1) = locs(i+1);
        pks_f(end+1) = pks(i+1);
    end
end

%��������� QRS
Fa = 13; %������� ��������� ��� ���������� ��� qrs(��������,����� ��������)
wName = 'bior1.5';
Fwt = centfrq(wName);
%scale = round((Fwt*Fs)/Fa);%�������������� ����� 
scale = 40;
wt1 = cwt(Sig,scale,wName); %���

len_wn = round(0.75*Fs/2); %�������� ������ ���� � �������� (���������� �� ����)
Q_pks = [];
R_pks = [];
S_pks = [];
QRS_l_bound = [];
QRS_r_bound = [];
delta_bound = round(0.035*Fs);
for i=1:length(locs_f)
    %��������� ������ ����� ������ ������ ����
    if locs_f(i)-len_wn < 0
        slice = wt1(1:locs_f(i)+len_wn);
        [m,ind_s]=max(slice);
        for j = ind_s:length(slice) %S ��� - ����������� � ����� ������ �� ���������
            if slice(j) < 0
                S_pks(end+1)=j;
                QRS_r_bound(end+1)=j+delta_bound;
                break
            end
        end

        for j = ind_s:-1:1 %R ���
            if slice(j) < 0
                R_pks(end+1)=j;
                break
            end
        end

        [m,ind_q]=min(slice(1:R_pks(end)));
        for j = ind_q:-1:1 %Q ���
            if slice(j) > 0
                Q_pks(end+1)=j;
                QRS_l_bound(end+1)=j-round(0.02*Fs);
                break
            end
        end
        continue
    end
    %---------------------------------------------
    if locs_f(i)+len_wn > length(wt1)
        slice = wt1(locs_f(i)-len_wn:end);
        [m,ind_s]=max(slice);
        for j = ind_s:length(slice) %S ��� - ����������� � ����� ������ �� ���������
            if slice(j) < 0
                S_pks(end+1)=j+locs_f(i)-len_wn;
                QRS_r_bound(end+1)=j+locs_f(i)-len_wn+delta_bound;
                break
            end
        end

        for j = ind_s:-1:1 %R ���
            if slice(j) < 0
                R_pks(end+1)=j+locs_f(i)-len_wn;
                break
            end
        end

        [m,ind_q]=min(slice(1:R_pks(end)-locs_f(i)+len_wn));
        for j = ind_q:-1:1 %Q ���
            if slice(j) > 0
                Q_pks(end+1)=j+locs_f(i)-len_wn;
                QRS_l_bound(end+1)=j+locs_f(i)-len_wn-round(0.02*Fs);
                break
            end
        end
        continue
    end
    %================================================
    slice = wt1(locs_f(i)-len_wn:locs_f(i)+len_wn);
    [m,ind_s]=max(slice);
    for j = ind_s:length(slice) %S ��� - ����������� � ����� ������ �� ���������
        if slice(j) < 0
            S_pks(end+1)=j+locs_f(i)-len_wn;
            QRS_r_bound(end+1)=j+locs_f(i)-len_wn+delta_bound;
            break
        end
    end
    
    for j = ind_s:-1:1 %R ���
        if slice(j) < 0
            R_pks(end+1)=j+locs_f(i)-len_wn;
            break
        end
    end
    
    [m,ind_q]=min(slice(1:R_pks(end)-locs_f(i)+len_wn));
    for j = ind_q:-1:1 %Q ���
        if slice(j) > 0
            Q_pks(end+1)=j+locs_f(i)-len_wn;
            QRS_l_bound(end+1)=j+locs_f(i)-len_wn-round(0.02*Fs);
            break
        end
    end
end

%�������� QRS � ������������
Sig_no_qrs = Sig;
for i = 1:length(QRS_l_bound)
    x_l = QRS_l_bound(i);
    x_r = QRS_r_bound(i);
    y_l = Sig(x_l);
    y_r = Sig(x_r);
    step = (y_r - y_l)/(x_r - x_l);
    for j = x_l:x_r
        Sig_no_qrs(j) = Sig_no_qrs(j-1) + step;
    end
end

%��������� P,T ����
Fa = 8; %������� ��������� ��� ���������� ��� qrs(��������,����� ��������)
wName = 'bior1.5';
Fwt = centfrq(wName);
scale = round((Fwt*Fs)/Fa);%�������������� ����� 
%scale = 200;
wt2 = cwt(Sig_no_qrs,scale,wName); %���

P_pks = [];
P_l_bound = [];
P_r_bound = [];
    
T_pks = [];
T_l_bound = [];
T_r_bound = [];

%�������� � P ������
for i = 1:length(R_pks)
    len_w_p = len_wn*0.9; %len_wn - (locs_f(i) - R_pks(i)); %���� ��� P �����
    slice_p = wt2(R_pks(i)-len_w_p:R_pks(i)); %P �����

    %���� min �� ������������ ���������� 0.095*Fs : QRS_l_bound(i) - (R_pks(i)-len_w_p)
    [m, ind_p] = min(slice_p((round(0.095*Fs)):round(QRS_l_bound(i) - (R_pks(i)-len_w_p))));
    ind_min = ind_p + round(0.095*Fs);
    threshold1 = m/2;
    flag = 0;
    for j = ind_min:length(slice_p) %P ��� �� ����������� � �����
        if slice_p(j) > 0
            P_pks(end+1)=j + R_pks(i) - len_w_p;
            break
        end
        if j == length(slice_p) && slice_p(j) < 0 %������ ����� ��� ����������� ����
            P_pks(end+1)=R_pks(i) - 0.16*Fs; %��������� ����� � ������� �� ��������
            flag = 1;
        end
    end
    
    if flag == 1
        P_l_bound(end+1)=P_pks(i) - 0.045*Fs;
        P_r_bound(end+1)=P_pks(i) + 0.045*Fs;
        continue
    end
    
    for j = ind_min:length(slice_p)-1 %������ ������� P ����� �� ������ ������� ���������
        if slice_p(j) > slice_p(j-1) && slice_p(j) > slice_p(j+1)
            P_r_bound(end+1)=j + R_pks(i) - len_w_p;
            break
        end
    end
    
    for j = ind_min:-1:2 %����� ������� P ����� �� �������� ��������
        if slice_p(j) > threshold1
            P_l_bound(end+1)=j + R_pks(i) - len_w_p;
            break
        end
    end
end

%�������� � T ������
for i = 1:length(R_pks)
    len_w_t = len_wn + (locs_f(i) - R_pks(i)); %���� ��� T �����
    slice_t = wt2(R_pks(i):R_pks(i)+len_w_t); %T �����
    flag = 0;
    
    [m, ind_t] = max(slice_t);
    threshold1 = m/2;
    for j = ind_t:length(slice_t)
        if slice_t(j) < threshold1
            T_r_bound(end+1) = j + R_pks(i); %������ ������� T ����� �� �������� ���������
            break
        end
        if j == length(slice_t) %���� ������� �� ������� ��������� ���������� ����������            
            T_r_bound(end+1) = R_pks(i) + round(0.22*Fs);
            T_pks(end+1) = R_pks(i) + round(0.145*Fs);
            T_l_bound(end+1) = R_pks(i) + round(0.07*Fs);
            flag = 1;
        end
    end
    
    if flag == 1
        continue
    end
    
    for j = ind_t:-1:1 %T ���
        if slice_t(j) < 0
            T_pks(end+1) = j + R_pks(i);
            break
        end
        if j == 1
            T_pks(end+1) = R_pks(i) + (2/3)*(T_r_bound(i)-R_pks(i));
        end
    end
    
    [m, ind_t] = min(slice_t);
    threshold2 = m/2;
    for j = ind_t:-1:1 %����� ������� � ����� �� �������� ��������
        if slice_t(j) > threshold2
            T_l_bound(end+1) = j + R_pks(i);
            break
        end
    end
end

axes(handles.axes1);
plot(t, Sig(i_start:i_stop), 'k');
title(file);
xlabel('t, c');
ylabel('����������, ��');

y_max_sig = max(Sig(i_start:i_stop))*1.2;
y_min_sig = min(Sig(i_start:i_stop))*1.2;
xlim([t(1) t(end)])
ylim([y_min_sig y_max_sig])

hold on

all_peaks = [P_pks; Q_pks; R_pks; S_pks; T_pks];

for i=1:[size(all_peaks)](1)
    ind_start = find(all_peaks(i,:)>=i_start,1,'first'); % ���������� ������ �����
    ind_stop = find(all_peaks(i,:)<=i_stop,1,'last');
    peaks_i = all_peaks(i, ind_start:ind_stop);
    peaks_i_t = peaks_i./Fs;
    for j=1:length(peaks_i)
          plot(peaks_i_t(j),Sig(peaks_i(j)),'o','MarkerSize',6);
    end
end

all_bounds = [P_l_bound; P_r_bound; QRS_l_bound; QRS_r_bound; T_l_bound; T_r_bound];

for i=1:[size(all_bounds)](1)
    ind_start = find(all_bounds(i,:)>=i_start,1,'first'); % ���������� ������ �����
    ind_stop = find(all_bounds(i,:)<=i_stop,1,'last');
    peaks_i = all_bounds(i, ind_start:ind_stop);
    peaks_i = peaks_i./Fs;
    for j=1:length(peaks_i)
          plot([peaks_i(j) peaks_i(j)],[y_min_sig y_max_sig],':k');
    end
end


%��������� ���
%������� ����� QRS
QRS_i = [];
for i=1:length(Q_pks)
    QRS_i(end+1) = (S_pks(i) - Q_pks(i))/Fs*1000;
end
QRS_ms_mean = round(mean(QRS_i));

T_i = [];
for i=1:length(T_l_bound)
    T_i(end+1) = (T_r_bound(i) - T_l_bound(i))/Fs*1000;
end
T_ms_mean = round(mean(T_i)); %����� 100 - 200

P_i = [];
for i=1:length(P_l_bound)
    P_i(end+1) = (P_r_bound(i) - P_l_bound(i))/Fs*1000;
end
P_ms_mean = round(mean(P_i)); %����� 70 - 110

PR_i = [];
for i=1:length(P_pks)
    PR_i(end+1) = (R_pks(i) - P_pks(i))/Fs*1000;
end
PR_ms_mean = round(mean(PR_i)); %����� 120 - 200

QT_i = [];
for i=1:length(T_pks)
    QT_i(end+1) = (T_pks(i) - Q_pks(i))/Fs*1000;
end
QT_ms_mean = round(mean(QT_i));

ST_i = [];
for i=1:length(T_pks)
    ST_i(end+1) = (T_pks(i) - S_pks(i))/Fs*1000;
end
ST_ms_mean = round(mean(ST_i)); %����� 60 - 150

%RR ��������� � ��������
RR = [];
for i=2:length(R_pks)
    RR(end+1) = (R_pks(i) - R_pks(i-1)) / Fs;
end

RR_mean = mean(RR); %������� RR

%��� ��� ������������ ��������� QT ������� �� ������� ���������� ����� 
%(��������� ��� ��� ����������), ��� ������ ��� ������ ���� ������������� 
%������������ ���. ���� ����� ������������ ������� �������
QT_c = QT_ms_mean/1000/sqrt(RR_mean); %RR_mean � ��������
QT_c_ms = round(QT_c*1000); %����� 300 - 450

%����� ������ � �������
pokazateli = [{QRS_ms_mean} {'60 - 100'};{T_ms_mean} {'100 - 200'};{P_ms_mean} {'70 - 110'};...
    {PR_ms_mean} {'120 - 200'};{QT_c_ms} {'320 - 440'};{ST_ms_mean} {'60 - 150'}];
set(handles.uitable1,'Data',pokazateli);
set(handles.uitable1,'FontSize',9);

%======================������ ���======================
%������� ��������� ����������
HR = round(1/RR_mean*60);
%����������� ����������
SDNN = std(RR);
%����������� ��������
CV = SDNN/RR_mean*100;
%������� ���������� ���������� > 50 ��
n_sd50 = 0;
for i=1:length(RR)-1
    sd = abs(RR(i+1) - RR(i))*1000;
    if sd > 50
        n_sd50 = n_sd50 + 1;
    end
end
PNN50 = n_sd50/length(RR)*100;


maxRR = max(RR);
maxRRp = maxRR*1.2;

%�����������
dH=0.05;    % ��� ����������� (50 ��) 
X=0:dH:maxRR; % ���������� �� ��� ������� (RR-��������, �) 
H=histc(RR,X); % ������ ����������� 
SH=sum(H);  % ����� ����������� (����� RR-����������) 
PH=H/SH*100; % ��������� ����������� � % 

maxh=max(PH)*1.2;

RRmin = min(RR);
RRmax = max(RR);
MxDMn = RRmax - RRmin; % ������������ ������ 
[AMo, iMo] = max(PH); % ��������� ���� � ������ ���� 
Mo = iMo*dH; % ����
SI = AMo/(2*Mo*MxDMn); % ������-������

%�������������
NRR=length(RR);                 % ����� ����� RR-���������� 

%������������ �����������
t = 0;
for i = 1:NRR
    t = t + RR(i);
    tRR(i) = t;
end
sRR = csaps(tRR, RR, 1);
Fd = 4;
T = 1/Fd;
tsRR = 0:T:tRR(NRR);
RR4Hz = ppval(sRR, tsRR);

%������ �������
nfft = 2048; %����� ����� ��� ��������� ��� 
%���������� ��������� ������ � ������� �������� ������� � ��: 
RR0 = detrend(RR4Hz)*1000; 
df = Fd/nfft; %��� �� ������� 
Fmax = 0.5*Fd; %������������ ������� ��� ������� 
Nf = fix(Fmax/df); %����� �������� �� ��� ������

window = hamming(length(RR4Hz)); % ���� ��������
[Pxx,f] = periodogram(RR0,window,nfft,Fd);

flim = [0.003 0.04 0.15 0.4];	% ������� ���������� ������
flim2 = round([0.003 0.04 0.15 0.4]./Fd.*length(Pxx).*2);

%������ �����������
VLF = 0;
LF = 0;
HF = 0;	% ��������� ��������
i = 0; % ���������� ����� � �������
f = 0;
while f <= flim(4)	% ������� ������ �� 0,4 �� 
    f = df*i;	% ������� �������
    i = i+1;	% ������ ������� ��� 
    if f >= flim(1) && f < flim(2)	% �������� VLF
        VLF = VLF + Pxx(i)*df;	% ������ VLF 
    elseif f >= flim(2) && f < flim(3) % �������� LF
        LF = LF + Pxx(i)*df;	% ������ LF 
    elseif f >= flim(3)	% �������� HF
        HF = HF + Pxx(i)*df;	% ������ HF
    end
end
VLF = round(VLF);
%�������� �������
TP = VLF + LF + HF;
%����������� LF/HF
LFHF = LF/HF;

%������ ����
%� - ��������� ������ ��������� [RR_mean]
if RR_mean < 0.66
    TER = 2;
elseif RR_mean >= 0.66 && RR_mean < 0.8
    TER = 1;
elseif RR_mean >= 0.8 && RR_mean < 1.0
    TER = 0;
elseif RR_mean >= 1.0 && RR_mean <= 1.2
    TER = -1;
else
    TER = -2;
end

%� - ������� ����������� [SDNN, MxDMn, CV]
if SDNN <= 0.02 && MxDMn <= 0.1*RR_mean && CV <= 2.0
    FA = 2;
elseif SDNN >= 0.1 && MxDMn > 0.3*RR_mean && CV > 8.0
    FA = 1;
elseif MxDMn >= 0.11*RR_mean && MxDMn <= 0.3*RR_mean
    FA = 0;
elseif MxDMn >= 0.45*RR_mean
    FA = -1;
elseif SDNN >= 0.11 && MxDMn > 0.6*RR_mean && CV > 8.0
    FA = -2;
end

%� - ������������ ��������� [MxDMn, AMo, SI]
if MxDMn < 0.06 && AMo > 80 && SI > 500
    VH = 2;
elseif MxDMn < 0.15 && AMo > 50 && SI > 200
    VH = 1;
elseif MxDMn >= 0.15 && MxDMn <= 0.3 && AMo <= 50 && AMo >= 30 && SI >= 50 && SI <= 200
    VH = 0;
elseif MxDMn > 0.3 && AMo < 30 && SI < 50
    VH = -1;
elseif MxDMn > 0.5 && AMo < 15 && SI < 25
    VH = -2;
end

%� - ������������ ��������� [CV]
if CV < 3.00
    SR = 2;
elseif CV >= 3.0 && CV <= 6.0
    SR = 0;
elseif CV > 6.0
    SR = -2;
end

%� - ���������� ����������� ������� ������� TP, LF, VLF, HF
coef1 = VLF/TP*100;
coef2 = LF/TP*100;
coef3 = HF/TP*100;
if coef1 > 70 && coef2 > 25 && coef3 < 5
    ASNC = 2;
elseif coef1 > 60 && coef3 < 20
    ASNC = 1;
elseif coef1 > 40 && coef1 <= 60 && coef3 >= 20 && coef3 <= 30 
    ASNC = 0;
elseif coef1 < 40 && coef3 > 30
    ASNC = -1;
elseif coef1 < 20 && coef3 > 40
    ASNC = -2;
else
    ASNC = 0;
end

PARS = abs(TER) + abs(FA) + abs(VH) + abs(SR) + abs(ASNC);

%����� ������ � �������
pokazateli_VSR = [{HR} {'60 - 90'};
    {round(SDNN*1000)} {'30 - 100'};
    {CV} {'3 - 12'};
    {PNN50} {' '};
    {round(TP)} {' '};
    {round(VLF)} {' '};
    {round(LF)} {' '};
    {round(HF)} {' '};
    {LFHF} {' '};
    {round(SI)} {'50 - 150'};
    {PARS} {'1 - 4'};];

set(handles.uitable2,'Data',pokazateli_VSR);
set(handles.uitable2,'FontSize',9);

%����������

conclusion = [{'�����.'};
    {'��������� �������������� ����������.'};
    {'���������� �������������� ����������.'};
    {'����� ���������� �������������� ���������� (�������������� ������������ ������).'};
    {'����������� (���������) ������������ ������.'};
    {'���� ���������.'}];

if PARS == 1 || PARS == 2
    set(handles.conclusion,'String',char(conclusion(1)));
elseif PARS == 3 || PARS == 4
    set(handles.conclusion,'String',char(conclusion(2)));
elseif PARS == 5 || PARS == 6
    set(handles.conclusion,'String',char(conclusion(3)));
elseif PARS == 7
    set(handles.conclusion,'String',char(conclusion(4)));
elseif PARS == 8
    set(handles.conclusion,'String',char(conclusion(5)));
else
    set(handles.conclusion,'String',char(conclusion(6)));
end


function slider2_Callback(hObject, eventdata, handles)

global Sig len_signal file all_peaks all_bounds;
slider_val = get(hObject,'Value');

Fs = get(handles.edit1,'string');
Fs = str2double(Fs); % ������� �������������


len_window = get(handles.edit2,'string'); % �������� ����� ����
len_window = str2double(len_window);

i_start = round((len_signal - len_window*Fs)*slider_val)+1;
i_stop = i_start + len_window*Fs - 1;

T = 1/Fs;
tmax = i_stop/Fs;
t = i_start/Fs:T:tmax;


axes(handles.axes1);
cla('reset');
plot(t, Sig(i_start:i_stop), 'k');
title(file);
xlabel('t, c');
ylabel('����������, ��');

y_max_sig = max(Sig(i_start:i_stop))*1.2;
y_min_sig = min(Sig(i_start:i_stop))*1.2;
xlim([t(1) t(end)])
ylim([y_min_sig y_max_sig])

hold on

for i=1:[size(all_peaks)](1)
    ind_start = find(all_peaks(i,:)>=i_start,1,'first'); % ���������� ������ �����
    ind_stop = find(all_peaks(i,:)<=i_stop,1,'last');
    peaks_i = all_peaks(i, ind_start:ind_stop);
    peaks_i_t = peaks_i./Fs;
    for j=1:length(peaks_i)
          plot(peaks_i_t(j),Sig(peaks_i(j)),'o','MarkerSize',6);
    end
end


for i=1:[size(all_bounds)](1)
    ind_start = find(all_bounds(i,:)>=i_start,1,'first'); % ���������� ������ �����
    ind_stop = find(all_bounds(i,:)<=i_stop,1,'last');
    peaks_i = all_bounds(i, ind_start:ind_stop);
    peaks_i = peaks_i./Fs;
    for j=1:length(peaks_i)
          plot([peaks_i(j) peaks_i(j)],[y_min_sig y_max_sig],':k');
    end
end


function slider2_CreateFcn(hObject, eventdata, handles)

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit2_Callback(hObject, eventdata, handles)



function edit2_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
global RR;
% ����� ����� RR-����������
NRR=length(RR);
t = 0;
for i = 1:NRR
    t = t + RR(i);
    tRR(i) = t;
end
sRR = csaps(tRR, RR, 1);
Fd = 4;
T = 1/Fd;
tsRR = 0:T:tRR(NRR);
RR4Hz = ppval(sRR, tsRR);

%������ �������
nfft = 2048; %����� ����� ��� ��������� ��� 
%���������� ��������� ������ � ������� �������� ������� � ��: 
RR0 = detrend(RR4Hz)*1000; 
df = Fd/nfft; %��� �� ������� 
Fmax = 0.5*Fd; %������������ ������� ��� ������� 
Nf = fix(Fmax/df); %����� �������� �� ��� ������

window = hamming(length(RR4Hz)); % ���� ��������
[Pxx,f] = periodogram(RR0,window,nfft,Fd);

axes(handles.axes2);
cla('reset');
hold on
title('������');
xlabel('��^2');
ylabel('��^2');

plot(f(1:Nf),Pxx(1:Nf));

flim = [0.003 0.04 0.15 0.4];	% ������� ���������� ������
flim2 = round([0.003 0.04 0.15 0.4]./Fd.*length(Pxx).*2);
area(f(flim2(1):flim2(2)), Pxx(flim2(1):flim2(2)), 'FaceColor', [0.1 1 0.1])

area(f(flim2(2):flim2(3)), Pxx(flim2(2):flim2(3)), 'FaceColor', [1 1 0.1])

area(f(flim2(3):flim2(4)), Pxx(flim2(3):flim2(4)), 'FaceColor', [1 0.1 0.1])

area(f(flim2(4):end), Pxx(flim2(4):end), 'FaceColor', [0.1 0.1 0.1])

xlim([0 1])
