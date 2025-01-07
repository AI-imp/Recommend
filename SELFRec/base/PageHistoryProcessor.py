import pandas as pd
import sys

class PageHistoryProcessor:
    def __init__(self, file,conf,read_for_dataset=False):
        self.read_for_dataset = read_for_dataset
        self.excel_file = file
        self.savedata = conf['savedata']
        self.savedir = conf['savedir']
        self.split = conf['split'].strip("'")


    def process_page_history(self):
        # 读取excel文件
        df = pd.read_excel(self.excel_file)  # 跳过标题行

        # 提取PagePath列中ProID后的数值
        df['PagePath'] = df['PagePath'].str.extract(self.split)

        # 删除PagePath为空的行
        df = df.dropna(subset=['PagePath'])

        # 保留Person列
        df = df[['Person', 'PagePath']]
        df = df.drop_duplicates()
        df['score'] = 1
        
        # 返回结果
        return df

    # 通过读取数据库进行划分
    def process_page_history2(self,df):
        # 读取excel文件
        df = df  # 跳过标题行
        # 提取PagePath列中ProID后的数值
        df['PagePath'] = df['PagePath'].str.extract(self.split)
        # 删除PagePath为空的行
        df = df.dropna(subset=['PagePath'])
        # 保留Person列
        df = df[['Person', 'PagePath']]
        df = df.drop_duplicates()
        df['score'] = 1
        # 返回结果
        return df

    def save_to_txt(self, result, output_file):
        # 将结果保存为txt文件
        result.to_csv(output_file, sep=' ', index=False, header=False)
        print(f"保存结果到{output_file}成功！")


if __name__ == '__main__':
    sys.path.append('/home/zzj/code/SELFRec')
    from util.settingconf import SettingConf
    conf = SettingConf('/home/zzj/code/SELFRec/setting/setting.conf')
    conf = conf.config['T_tianRecommended']
    processor = PageHistoryProcessor('/home/zzj/code/dataset/priData/PageHistory.xlsx',conf)
    result = processor.process_page_history()
    processor.save_to_txt(result, 'out.txt')

