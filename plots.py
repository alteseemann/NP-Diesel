from tkinter import *
from tkinter import filedialog as fd
import pandas
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import traceback
try:
	repeat = 'y'
	filename = ''
	while (repeat != 'n' or repeat != 'nox'):
		root      = Tk()
		root.withdraw()
		filename  = fd.askopenfilename()
		root.destroy()

		excel_data = pandas.read_excel(filename,engine='openpyxl',index_col=0)
		p = []
		fi = []
		fi0 = excel_data.columns.ravel()#список заголовков столбцов таблицы, т.е. нулевая строка, начиная с ячейки 1(строка А, начиная со столбца В)
		for f in fi0:
			f_fl = float(f)
			fi.append(float('{:.3f}'.format(f_fl)))
			p.append(float(excel_data[f].tolist()[0]))

		f_min = float(input('Граница слева: '))
		f_max = float(input('Граница справа: '))
		n = float(input('Количество оборотов в минуту : '))
		w = float(input('Объемная доля воды: '))
		h_f = 5
		h_p = 1

		fi_plot = []
		p_plot  = []
		for f in range(int(f_min),int(f_max)):
			pos = fi.index(f)
			fi_plot.append(fi[pos])
			p_plot.append(p[pos])

		#Строим график
		fig, x = plt.subplots()
		#добавляем данные графика
		x.plot(fi_plot,p_plot,color='red',linewidth = 2,label='P(град. п.к.в.)')
		#Устанавливаем интервал основных и вспомогательных делений:
		x.xaxis.set_major_locator(ticker.MultipleLocator(h_f))
		x.xaxis.set_minor_locator(ticker.MultipleLocator(0.5*h_f))
		x.yaxis.set_major_locator(ticker.MultipleLocator(h_p))
		x.yaxis.set_minor_locator(ticker.MultipleLocator(0.5*h_p))
		#  Добавляем линии основной сетки:
		x.grid(which='major',color = 'k')
		#  Включаем видимость вспомогательных делений:
		x.minorticks_on()
		#  Теперь можем отдельно задавать внешний вид вспомогательной сетки:
		x.grid(which='minor',color = 'gray',linestyle = ':')
		#Размер окна графика
		fig.set_figwidth(18)
		fig.set_figheight(12)
		#добавляем подписи к осям:
		x.set_xlabel('град. п.к.в.',fontsize = 15,color = 'black')
		x.set_ylabel('P, атм',fontsize = 15,color = 'black')
		#Заголовок графика
		title = 'Индикаторная диаграмма: '+str(n)+' об/мин. '+str(w)+' % воды'
		x.set_title(title,fontsize = 15,color = 'black',pad=10)
		#Помещаем легенду на график
		x.legend(shadow = True,fontsize = 15)
		
		name = 'plots/diagrams/'+str(n)+'_'+str(w)+'.png'

		fig.savefig(name)

		plt.show()

		repeat = input('Построить новый график?(y) Или перейти в режим NOx? (nox): ')

except Exception as e:
	print (e.__class__.__name__)
	print (e.args)
	print (traceback.format_exc())
	input()