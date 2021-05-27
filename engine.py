import eel
from tkinter import *
from sympy import *
import math
from tkinter import filedialog as fd
import pandas
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import special
from scipy.optimize import fsolve
from scipy import interpolate
import os



eel.init('web',allowed_extensions=['.js', '.html'])

def open_diagram(filename):# получаем данные индикаторной диаграммы из файла экселя
	excel_data = pandas.read_excel(filename,engine='openpyxl',index_col=0)
	p = []
	fi = []
	fi0 = excel_data.columns.ravel()#список заголовков столбцов таблицы, т.е. нулевая строка, начиная с ячейки 1(строка А, начиная со столбца В)
	for f in fi0:
		f_fl = float(f)
		fi.append(float('{:.3f}'.format(f_fl)))
		p.append(float(excel_data[f].tolist()[0]))# получить данные из столбца и преобразовать их в список значений. Поскольку каждому углу соответствует одно давление, берем 0 элемент списка
	return p,fi


def open_constants(filename): #Открывает файл с константами и преобразует его в именованный список
	excel_data = pandas.read_excel(filename,engine='openpyxl',index_col=0)
	#print (excel_data)
	elements = []
	temperatures = []
	constants = {}
	#проходка по таблице и преобразование ее в массив
	for i in excel_data:
		#массив названий столбцов
		temperatures.append(i)
	for j in excel_data.index:
		#массив названий строк
		elements.append(j)
	for element in elements:
		constants[element] = {}
		for temperature in temperatures:
			constants[element][temperature] = excel_data[temperature][element]
	#сформирован список вида {element1:{t1:const1...tn:constn}...elementn:{t1:const1...tn:constn}}
	return constants,temperatures,elements

@eel.expose()
def get_filename(file_type):#функция открывает диалоговое окно выбора файла, после чего передает полное имя файла в JS
	root      = Tk()
	root.withdraw()
	filename  = fd.askopenfilename()
	name_ar   = filename.split('/')
	full_name = name_ar[-1]
	name      = full_name.split('.')[0]
	extension = full_name.split('.')[1]
	eel.getFile(filename,file_type)
	root.destroy()

def get_initial_data(raw_data):
	#p -массив давлений, атм
	#fi - массив углов, градусы
	raw             = {}
	#определяем путь к экселевским файлам
	diagrams_excel  = raw_data['diagrams_excel']
	constants_excel = raw_data['constants_excel']
	#Получаем точечно заданную зависимость давления (атм) от угла пкв (град) и формируем исходные данные для расчета
	p,fi            = open_diagram(diagrams_excel)
	#Задаемся исходными данными для расчета
	scale           = int(raw_data['diagram_scale'])
	rpm             = float(raw_data['rpm'])
	fuel_flow       = float(raw_data['fuel_flow'])
	air_flow        = float(raw_data['air_flow'])
	fuel_type       = int(raw_data['fuel_type'])
	engine_type     = int(raw_data['engine_type'])
	w               = float(raw_data['water_percent'])/100
	#Перевод объемной доли воды w в отношение массовых расходов воды и топлива
	
	if (fuel_type == 1):
		k = (1000*w)/(860*(1-w))#отношение расхода чистой воды к расходу чистого топлива
		r_w = (1000*w)/(1000*w+860*(w-1))#массовая доля воды в топливе
	if (fuel_type == 2):
		k = (1000*w)/(840*(1-w))
		r_w = (1000*w)/(1000*w+840*(w-1))
		
	#Геометрия и размеры двигателя
	l               = 140-39/2-21/2 #длина шатуна,мм
	r               = 36# радиус кривошипа, мм
	n_nom           = 3000#об/мин, номинальная частота
	raw['la']       = r/l#относительная длина шатуна
	raw['vh']       = 0.296#рабочий объем цилиндра,л
	raw['e']        = 19#степень сжатия
	raw['v0']       = raw['vh']/(raw['e']-1)#объем камеры сгорания,л
	raw['s']        = 0.062#м, ход поршня
	raw['d']        = 0.078#м, диаметр цилиндра
	raw['al']       = 220#Вт/м*К, теплопроводность алюминия  - материала ЦПГ ДВС
	raw['ribs']     = 15#Коэффициент оребрения цилиндра, отношение внешней площади к внутренней
	raw['delta']    = 200#1/м, величина, обратная толщине стенки цилиндра(1/d)
	omega           = rpm*0.1047#угловая скорость, рад/с
	
	
	raw['qc']       = ((4*fuel_flow)/(120*rpm))*(1-r_w)#цикловая подача топлива, кг/цикл
	raw['a']        = air_flow/(14.45*fuel_flow*(1-r_w))#коэффициент избытка воздуха
	raw['m0']       = raw['qc']*((1-r_w)/172 + r_w/18 + 14.45*raw['a']/29)#кол-во вещ-ва в цилиндре, кмоль
	raw['q0']       = 43000*(1-0.125*k)#Низшая теплота сгорания, кДж/кг
	raw['w']        = k
	raw['w0']       = w
	raw['r']        = r
	raw['t_engine'] = float(raw_data['cylinder_temperature'])+273#К, температура головки цилиндров
	raw['omega']    = omega
	
	#Параметры окружающей среды
	raw['t_air']    = float(raw_data['air_temperature'])+273#К, температура окружающего воздуха
	raw['p_air']    = float(raw_data['air_pressure'])*0.000133#МПа, давление окружающего воздуха
	raw['r_air']    = 3480*(raw['p_air']/raw['t_air'])#кг/м3 Плотность воздуха на впуске

	#Параметры наполнения цилиндра
	raw['nv']       = (33.6*air_flow)/(raw['vh']*rpm*raw['r_air'])#коэффициент наполнения
	raw['tr']       = 1450/raw['e'] + 1029/raw['a'] + 0.14*rpm - 494#температура остаточных газов
	prn             = 1.1*raw['p_air']
	ar              = (prn-1.035*raw['p_air'])*(10**8)*(1/(raw['p_air']*n_nom**2))
	raw['pr']       = raw['p_air']*(1.035+ar*(10**-8)*rpm**2)
	raw['gamma']    = (raw['pr']*raw['t_air'])/((raw['e']-1)*raw['tr']*raw['p_air']*raw['nv'])#коэффициент остаточных газов


	p_fi            = {}#Па, град
	v_fi            = {}#м3, град
	p_v             = {}#Па, м3
	diagram         = {}

	
	if (scale>0):#Удаляет scale элементов из начала массива давлений и из конца массива углов
		p_scale=[]
		fi1 = fi[::-1]
		fi_scale = []
		for i in range (scale,len(p)):
			p_scale.append(p[i])
			fi_scale.append(fi1[i])
			#fi_scale.append(fi[i])
		p = p_scale
		#fi = fi_scale
		fi = fi_scale[::-1]
	

	for i in range (0,len(fi)):
		fi_i = fi[i]*0.017
		v = (raw['v0']+0.5*raw['vh']*(1+1/raw['la']-cos(fi_i)-(1/raw['la'])*(1-(raw['la']**2)*(sin(fi_i))**2)**0.5))*10**(-3)
		p_i = p[i]*100000
		p_fi[fi[i]]= p_i
		v_fi[fi[i]] = v
		p_v[v] = p_i
		diagram[fi[i]] = [p[i],p_i,v]
	
	return p,fi,p_fi,v_fi,p_v,diagram,raw

def cvm(t,w,x,a,g = 0.03):
	#Определение средней молярной теплоемкости смеси при температуре t, отношении массы воды к массе топлива в эмульсии w, доле сгоревшего топлива х, коэф-те избытка в-ха а
	cv_steam = 22.955+573.8*(10**(-5))*t#теплоемкость пара,кДж/кмоль*К
	cv_air = 20.093+242.8*(10**(-5))*t#теплоемкость воздуха,кДж/кмоль*К
	cv_comprod = 21.876+334.9*(10**(-5))*t#теплоемкость продуктов сгорания,кДж/кмоль*К
	cvm = ((1.036*x+g)*cv_comprod + (a*(1+g)-(x+g))*cv_air + (w*x*cv_steam)/(8.946*(1-w)))/(0.0636*x + (1+g)*a + (w*x)/(8.946*(1-w)))
	return cvm

def combustion_index(fi,p_v,raw):
	#определяет степень выгорания топлива
	x_ar=[]
	x_par=[]
	fi_ar=[]
	l_ar = []
	p0 = list(p_v.values())[0]/1000
	v0 = list(p_v.keys())[0]
	pv0 = p0*v0
	x0 = 0
	#определяет степень выгорания в каждый момент времени 
	for key,value in p_v.items():	
		pv = key*value/1000#потенциальная функция, kдж
		dpv = pv-pv0
		t = pv/(8314*raw['m0'])#приблизительное значение температуры
		#определение работы совершенной газами в данный момент времени
		l = 0#Интегрируем каждый раз от начала горения, когда l = 0
		for key1,value1 in p_v.items():
			p_b = value1#верхний предел интегрирования
			v_b = key1
			if (v_b==list(p_v.keys())[0]):
				v_a = v_b
				p_a = p_b
			if (v_b!=list(p_v.keys())[0]):
				v_a = [k for k,v in p_v.items() if k<v_b][-1]
				p_a = p_v[v_a]
			dl= 0.5*(p_a+p_b)*(v_b-v_a)*0.001#работа,кДж = площадь под графиком(площадь трапеции)
			l+=dl#интеграл методом трапеций
			if (v_b==key):#Цикл прерывается, когда график проинтегрирован до текущего значения v = key(верхний предел равен текущему значению)
				break
		#print ('p0 = {}, p1 = {}, v0 = {}, v1 = {}, l = {}, l(n) = {}'.format(p0,p1,v0,v1,l,l_n))
		l_ar.append(l)
		#определение степени выгорания топлива
		a0 = raw['w']/(1-raw['w'])
		#a1 = 20.093 + 242.8*(10**(-5))*t
		a1 = cvm(t,raw['w'],x0,raw['a'],g=raw['gamma'])
		a2 = 1.89 + 0.305*a0 + (97.7+35.238*a0)*(10**(-5))*t
		a3 = 1/(0.94*(raw['a']+0.0606+0.1065*a0))
		x  = (a1*dpv+l*8.314)/(8.314*raw['qc']*raw['q0']-a2*a3*pv)
		x0 = x
		x_ar.append(x)
		if (x>=1):
			break
	#Массив значений углов (значения по оси х)
	for i in range(0,len(x_ar)):
		fi_ar.append(fi[i])
	#аппроксимирует зависимость формулой Вибе
	fim = int((fi_ar[0]+fi_ar[-1])/2)
	xm = x_ar[fi_ar.index(fim)] 
	fi_z = fi_ar[-1]
	m = ln(-ln(1-xm)/6.908)/ln(fim/fi_z)-1#показатель сгорания
	for i in fi_ar:
		x_par.append(1-math.exp(-6.908*(i/fi_z)**(m+1)))

	return x_ar,x_par,fi_ar,l_ar

#Определение осредненного по объему цилиндра заначения температуры в произвольный момент сгорания
def temperature(t_2,x_ar,raw,p):
	t_ar = []
	for x in x_ar:
		b = 1+x*(0.569+0.431*raw['w'])/(8.946*raw['a']*(1-raw['w'])*(1+raw['gamma']))#Текущее значение коэффициента молекулярного изменения
		e = 0.85
		dq = x*raw['q0']*raw['qc']#кДж
		bg = 1/(raw['gamma']+b)
		la23 = max(p)/p[0]
		t_z = 0
		t_0 = 900#первое приближение Т в методе итераций
		while (abs(t_z-t_0)>10):
			cv=cvm(t_0,raw['w'],x,raw['a'],g=raw['gamma'])
			cv_s = 22.955+573.8*(10**(-5))*t_0#теплоемкость пара,кДж/кмоль*К
			cv_a = 20.093+242.8*(10**(-5))*t_0#теплоемкость воздуха,кДж/кмоль*К
			cv_c = 21.876+334.9*(10**(-5))*t_0#теплоемкость продуктов сгорания,кДж/кмоль*К
			a1 = raw['m0']*t_2*(cv_a+raw['gamma']*cv_c+cv_s+(1+raw['gamma'])*la23*8.314)
			t_z = t_0
			t_0 = bg*(dq+a1)/(raw['m0']*(cv+8.314))
		t_ar.append(t_z)
	return (t_ar)


def linear_approximation(terms,k_dis,t_real):#Получает значения констант диссоциации при заданной температуре для каждого элемента путем линейной аппроксимации
	#terms - массив температур
	#k_dis=[] - исходный массив вида {element1:{t1:const1...tn:constn}...elementn:{t1:const1...tn:constn}}
	#t_real - температура, при которой нужно найти значение константы
	k={} #конечный список констант каждого элемента при данной температуре t_real
	#a = {}
	b = 100
	#c = {}
	#Определяем интервал, в котором находится значение температуры
	for t in terms:
		if (t_real>=t and t_real<(t+100)):
			t_max = t+100
			t_min = t
			if (t == terms[-1]):
				t_max = t
				t_min = t-100		
	#print (t_max,' ',t_min)	
	#На найденном интервале температур для каждого элемента линейно аппроксимируем константу, находим ее значение при t_real
	for element in k_dis:
		#print (k_dis[element][2500])
		#kdis[element][t] - значение логарифма констатнты данного елемента при данной температуре
		ar = k_dis[element]
		log_k = (t_real-t_min)*(ar[t_max]-ar[t_min])/(t_max-t_min) + ar[t_min]
		k[element] = 10**log_k	
	return k

def eq_comp(t,p0,constants_excel):#Определяет равновесный состав продуктов сгорания
	def system(p,x,pressure):#Основная система уравнений для определения равновесного состава
		a,b,c,d = p #a - H2, b - H2O, c - CO2, d - N2
		alpha = 0.36
		beta = 0.266
		gamma = 0.512
		dalton = x[1]*b/a + x[2]*(b**2)/(a**2) + x[3]*(b**3)/(a**3) + x[4]*a**0.5 + a + x[5]*b/(a**0.5) + b + x[6]*c*(a**2)/(b**2) + x[7]*c*a/b + c + x[8]*c*(a**4)/(b**2) + x[9]*d**0.5 + d + x[10]*b*(d**0.5)/a + x[11]*(b**2)*(d**0.5)/(a**2) + x[12]*(d**0.5)*(a**1.5) + x[13]*(b**1.5)*(d**0.5)/a + x[14]*c*(a**2.5)*(d**0.5)/(b**2) - pressure
		ballance1 = alpha * (x[1]*b/a + 2*x[2]*(b**2)/(a**2) + 3*x[3]*(b**3)/(a**3) + x[5]*b/(a**0.5) + b + 3*x[13]*(b**1.5)*(d**0.5)/a + x[7]*c*a/b + 2*c + 2*x[11]*(b**2)*(d**0.5)/(a**2) + x[10]*b*(d**0.5)/a) - x[6]*c*(a**2)/(b**2) - x[7]*c*a/b - c - x[8]*c*(a**4)/(b**2) - x[14]*c*(a**2.5)*(d**0.5)/(b**2)
		ballance2 = beta * (x[11]*(b**2)*(d**0.5)/(a**2) + x[9]*d**0.5 + 2*d + x[10]*b*(d**0.5)/a + x[12]*(d**0.5)*(a**1.5) + x[13]*(b**1.5)*(d**0.5)/a + x[14]*c*(a**2.5)*(d**0.5)/(b**2)) - x[1]*b/a - 2*x[2]*(b**2)/(a**2) - 3*x[3]*(b**3)/(a**3) - x[5]*b/(a**0.5) - b - 3*x[13]*(b**1.5)*(d**0.5)/a - x[7]*c*a/b - 2*c - 2*x[11]*(b**2)*(d**0.5)/(a**2) - x[10]*b*(d**0.5)/a 
		ballance3 = gamma * (x[4]*a**0.5 + 2*a + x[5]*b/(a**0.5) + 2*b + 4*x[8]*c*(a**4)/(b**2) + 3*x[12]*(d**0.5)*(a**1.5) + x[13]*(b**1.5)*(d**0.5)/a + x[14]*c*(a**2.5)*(d**0.5)/(b**2)) - x[6]*c*(a**2)/(b**2) - x[7]*c*a/b - c - x[8]*c*(a**4)/(b**2) - x[14]*c*(a**2.5)*(d**0.5)/(b**2)
		system = (dalton,ballance1,ballance2,ballance3)
		return system
	#принимает температуру, давление, путь к файлу эксель с константами диссоциации
	constants,temperatures,elements = open_constants(constants_excel)#открывает файл констант
	el_constants = linear_approximation(temperatures,constants,t)#расчитывает значения констант для каждого элемиента [element:k[T]]
	#Выражаем константы равновесия реакций через константы диссоциации элементов
	elc = el_constants 
	k = {}
	#a - H2, b - H2O, c - CO2, d - N2
	k[1]  = elc['O2']**0.5
	k[2]  = (elc['O2']**1.5)/elc['O3']
	k[3]  = elc['H2']**0.5
	k[4]  = (elc['H2O']**2)/(elc['O2']*elc['H2']**2)
	k[5]  = (elc['O2']**0.5 * elc['H2']**0.5)/elc['OH-']
	k[6]  = elc['CO2']/elc['O2']
	k[7]  = elc['CO']/elc['O2']**0.5
	k[8]  = elc['CH4']/elc['H2']**2
	k[9]  = elc['N2']**0.5
	k[10] = elc['NO']
	k[11] = elc['NO2']/(elc['O2']**0.5 * elc['NO'])
	k[12] = elc['NH3']/(elc['N2']**0.5 * elc['H2']**1.5)
	k[13] = (elc['HNO3']**2)/(elc['NO']**2 * elc['H2O'])
	k[14] = elc['HCN']/(elc['N2']**0.5 * elc['H2']**0.5)
	#коэффициенты при парциальных давлениях (когда выражаем все через a,b,c,d) в системе из 4х нелинейных уравнений
	x = {}
	x[1]  = k[1] * k[4]**0.5
	x[2]  = k[4]
	x[3]  = k[2] * k[4]**1.5
	x[4]  = k[3]
	x[5]  = k[5] * k[4]**0.6
	x[6]  = k[6] / k[4]
	x[7]  = k[6] / (k[7] * k[4]**0.5)
	x[8]  = k[6] / (k[4] * k[8])
	x[9]  = k[9]
	x[10] = k[1] * k[4]**0.5 * k[9] / k[10]
	x[11] = k[1] * k[4] * k[9] / (k[10] * k[11])
	x[12] = 1 / k[12]
	x[13] = k[1] * k[4]**0.5 * k[9] / (k[10] * k[13]**0.5)
	x[14] = k[6] / (k[4] * k[14])
	#система из 3х уравнений материального баланса + закон Дальтона
	p = 0.00001*p0
	a,b,c,d = fsolve(system,(0.01*p,0.1*p,0.1*p,0.7*p),args=(x,p))
	#print (ans)
	#Парциальные давления компонентов
	p_ar={}
	p_ar['O']    = x[1]*b/a
	p_ar['O2']   = x[2]*(b**2)/(a**2)
	p_ar['O3']   = x[3]*(b**3)/(a**3)
	p_ar['H']    = x[4]*a**0.5
	p_ar['H2']   = a
	p_ar['OH-']  = x[5]*b/(a**0.5)
	p_ar['H2O']  = b
	p_ar['C']    = x[6]*c*(a**2)/(b**2)
	p_ar['CO']   = x[7]*c*a/b
	p_ar['CO2']  = c
	p_ar['CH4']  = x[8]*c*(a**4)/(b**2)
	p_ar['N']    = x[9]*d**0.5
	p_ar['N2']   = d
	p_ar['NO']   = x[10]*b*(d**0.5)/a
	p_ar['NO2']  = x[11]*(b**2)*(d**0.5)/(a**2)
	p_ar['NH3']  = x[12]*(d**0.5)*(a**1.5)
	p_ar['HNO3'] = x[13]*(b**1.5)*(d**0.5)/a
	p_ar['HCN']  = x[14]*c*(a**2.5)*(d**0.5)/(b**2)
	#Равновесные объемные концентрации веществ
	r={}
	r['O']    = p_ar['O']/p
	r['O2']   = p_ar['O2']/p
	r['O3']   = p_ar['O3']/p
	r['H']    = p_ar['H']/p
	r['H2']   = p_ar['H2']/p
	r['OH-']  = p_ar['OH-']/p
	r['H2O']  = p_ar['H2O']/p
	r['C']    = p_ar['C']/p
	r['CO']   = p_ar['CO']/p
	r['CO2']  = p_ar['CO2']/p
	r['CH4']  = p_ar['CH4']/p
	r['N']    = p_ar['N']/p
	r['N2']   = p_ar['N2']/p
	r['NO']   = p_ar['NO']/p
	r['NO2']  = p_ar['NO2']/p
	r['NH3']  = p_ar['NH3']/p
	r['HNO3'] = p_ar['HNO3']/p
	r['HCN']  = p_ar['HCN']/p
	#for element in elements:
		#print ('P',element,' = ',p_ar[element])
		#print ('r',element,' = ',r[element])
	return r

def dno_dfi(x_ar,fi_ar,p_fi,t_ar,constants_excel,raw):#Определение содержания NO, путем интегрирования уравнения методом Рунге-Кутта
	def f(t,p_i,r,no_0,omega):
		f_res = (2.333*(10**(7))*p_i*math.exp(-(38020/t))*r['N2']*r['O']*(1-(no_0/r['NO'])**2))/(omega*8.314*t*(1+(2346/t)*math.exp(3365/t)*(no_0/r['O2'])))
		return f_res
	no_0 = 0
	h = 1
	no_ar = []
	for fi in fi_ar:
		#Параметры рабочего тела в начале, середине и конце рассматриваемого интервала
		if (fi==list(p_fi.keys())[0]):
			fi0 = fi
			fi1 = fi
		if (fi!=list(p_fi.keys())[0]):
			fi0 = [k for k,v in p_fi.items() if k<fi][-1]
			fi1 = fi
		
		x = x_ar[fi_ar.index(fi1)]
		b = 1+x*(0.569+0.431*raw['w'])/(8.946*raw['a']*(1-raw['w'])*(1+raw['gamma']))#Коэффициент молекулярного изменения
		m_cp = 0.498*raw['qc']*raw['a']*(1.0636*x+raw['gamma'])#Количество чистых продуктов сгорания, кмоль
		m = raw['m0']*(1+x*(b-1))#Количество вещества в цилиндре, кмоль
		m_0  = m-m_cp#Количество свежей смеси, кмоль
		r = m_cp/m#доля продуктов сгорания в заряде цилиндра

		p0 = p_fi[fi0]
		p1 = p_fi[fi1]
		t0 = t_ar[fi_ar.index(fi0)]
		t1 = t_ar[fi_ar.index(fi1)]
		p05 = (p0+p1)/2
		t05 = (t0+t1)/2
		r0 = eq_comp(t0,p0,constants_excel)
		r1 = eq_comp(t1,p1,constants_excel)
		r05 = eq_comp(t05,p05,constants_excel)

		#Численное интегрирование
		k1 = f(t0,p0,r0,no_0,raw['omega'])
		no1 = no_0+h*k1/2
		k2 = f(t05,p05,r05,no1,raw['omega'])
		no2 = no_0+h*k2/2
		k3 = f(t05,p05,r05,no2,raw['omega'])
		no3 = no_0+h*k3
		k4 =f(t1,p1,r1,no3,raw['omega'])
		dno = (h/6)*(k1+2*k2+2*k3+k4)
		if (fi==list(p_fi.keys())[0]):
			dno = 0
		no_0+=dno
		no_ar.append(no_0*(10**6)*r)
		
	return no_ar


@eel.expose()
def main(json_data):
	
	raw_data = json.loads(json_data)#декодирование json
	#Преобразуем исходные данные
	p,fi,p_fi,v_fi,p_v,diagram,raw = get_initial_data(raw_data)
	#print (fi)
	
	#Расчет процесса сжатия, в результате которого определяется температура сжатия t_0
	t_0 = float(raw_data['t0'])+273

	
	#Расчет процесса сгорания, определение зависимостей температуры в цилиндре и степени выгорания топлива от угла пкв
	x_ar,x_par,fi_ar,l_ar = combustion_index(fi,p_v,raw)
	#Расчет средней температуры рабочего тела
	t_ar = temperature(t_0,x_par,raw,p)
	#print (t_ar)
	
	#Расчет на токсичность
	constants_excel = raw_data['constants_excel']
	no = dno_dfi(x_par,fi_ar,p_fi,t_ar,constants_excel,raw)
	#print (no)
	
	
	#Строим график степени выгорания топлива от угла пкв
	fig, x = plt.subplots()
	#добавляем данные графика
	x.plot(fi_ar,x_ar,linestyle=':',color='black',linewidth = 4,label='Эксперимент')
	x.plot(fi_ar,x_par,color='red',linewidth = 2,label='Формула Вибе')
	#Устанавливаем интервал основных и вспомогательных делений:
	x.xaxis.set_major_locator(ticker.MultipleLocator(1))
	x.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
	x.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
	x.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
	#  Добавляем линии основной сетки:
	x.grid(which='major',color = 'k')
	#  Включаем видимость вспомогательных делений:
	x.minorticks_on()
	#  Теперь можем отдельно задавать внешний вид вспомогательной сетки:
	x.grid(which='minor',color = 'gray',linestyle = ':')
	#Размер окна графика
	fig.set_figwidth(12)
	fig.set_figheight(8)
	#добавляем подписи к осям:
	x.set_xlabel('град. п.к.в.',fontsize = 15,color = 'black')
	x.set_ylabel('x',fontsize = 15,color = 'black')
	#Заголовок графика
	nox = 'Содержание NOx в ОГ - '+str(float('{:.0f}'.format(no[-1])))+' ppm. График х(f):'
	x.set_title(nox,fontsize = 20,color = 'red',pad=10)
	#Помещаем легенду на график
	x.legend(shadow = True,fontsize = 15)
	#отображаем график
	#plt.show()
	#сохраняем график
	#Проверяем, есть ли на диске файл с таким именем
	check = True
	i = 0
	name = 'plots/combustion_index_.png'
	while check:
		name = 'plots/combustion_index_'+raw_data['rpm']+'_'+raw_data['water_percent']+'_'+str(i)+'.png'
		check = os.path.exists(name)
		i+=1		
	fig.savefig(name)
	#plot_name = os.path.abspath('web/plots/combustion_index.png')
	#print (plot_name)
	#Отображаем график в главном окне программы
	plt.show()

eel.start('index.html',size=(800,650))