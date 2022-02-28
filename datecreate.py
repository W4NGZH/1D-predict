import datetime

def create_assist_date(datestart = None,dateend = None):
	# 创建日期辅助表

	if datestart is None:
		datestart = '2016-01-01'
	if dateend is None:
		dateend = datetime.datetime.now().strftime('%Y-%m-%d')

	# 转为日期格式
	datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
	dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
	date_list = []
	date_list.append(datestart.strftime('%Y-%m-%d'))
	while datestart<dateend:
# 日期叠加一天
	    datestart+=datetime.timedelta(days=+3)
# 日期转字符串存入列表
	    date_list.append(datestart.strftime('%Y-%m-%d'))
	print(date_list)



if __name__ == '__main__':
	create_assist_date("2003-07-10",'2007-07-10')
