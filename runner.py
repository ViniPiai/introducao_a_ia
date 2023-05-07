import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from pprint import pprint


class Runner:

    def __init__(self):
        self.df = pd.read_csv('data.csv', sep=',', encoding='utf8', header=0,
                              names=['name', 'tag', 'timestamp', 'date', 'x_coord', 'y_coord', 'z_coord', 'activity'])

    def run(self):
        print(self.df['x_coord'].head())
        self.process_string_column_to_int()
        self.add_time_from_date()
        self.see_distribution_of_coordinates()
        self.train()

    def add_time_from_date(self):
        # transforming Datetime Column to Date_time format
        self.df['date'] = pd.to_datetime(self.df['date'], format="%d.%m.%Y %H:%M:%S:%f")
        self.df['hour'] = self.df['date'].dt.hour
        self.df['minute'] = self.df['date'].dt.minute
        self.df['second'] = self.df['date'].dt.second
        self.df['microsecond'] = self.df['date'].dt.microsecond
        del self.df['date']  # Removing Date_time Column

    def see_distribution_of_coordinates(self):
        # Show plot of distribution of the coordinates x, y, z
        print(self.df['x_coord'].head())
        for i in self.df.select_dtypes(['float64']):
            sns.displot(self.df[i])
            plt.show()

        # Show 3D plot of the coordinates x, y, z
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.df['x_coord'], self.df['y_coord'], self.df['z_coord'], c=self.df['z_coord'], cmap='Greens')
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        ax.set_zlabel('Coordenada Z')
        plt.show()
        plt.figure(figsize=(20, 12))
        # Show heatmap
        sns.heatmap(self.df.corr(), annot=True)
        plt.show()

    def process_string_column_to_int(self):
        name_values = ["A01", "A02", "A03", "A04", "A05", "B01", "B02", "B03", "B04", "B05", "C01", "C02",
                       "C03", "C04", "C05", "D01", "D02", "D03", "D04", "D05", "E01", "E02", "E03", "E04", "E05"]
        tag_values = ["010-000-024-033", "020-000-033-111", "020-000-032-221", "010-000-030-096"]  #4
        activity_values = ['walking',
                           'falling',
                           'lying down',
                           'lying',
                           'sitting down',
                           'sitting',
                           'standing up from lying',
                           'on all fours',
                           'sitting on the ground',
                           'standing up from sitting',
                           'standing up from sitting on the ground'
                           ] # 11
        name = dict([(x, i) for i, x in enumerate(name_values)])
        tag = dict([(x, i) for i, x in enumerate(tag_values)])
        activity = dict([(x, i) for i, x in enumerate(activity_values)])
        self.df = self.df.replace({'tag': tag, 'name': name, 'activity': activity})

    def train(self):
        input_values = self.df.drop(['activity'], axis=1)
        target = self.df['activity']
        sm = SMOTE(sampling_strategy='minority', random_state=7)
        over_sampled_train_x, over_sampled_train_y = sm.fit_resample(input_values, target)
        over_sampled_train = pd.concat([pd.DataFrame(over_sampled_train_y), pd.DataFrame(over_sampled_train_x)], axis=1)
        over_sampled_train.columns = self.df.columns
        print(over_sampled_train.columns)
        pprint(over_sampled_train.values.tolist()[0])
        self.df = over_sampled_train
        print(over_sampled_train.shape)
        self.see_distribution_of_coordinates()

        scaler = RobustScaler()
        x = scaler.fit_transform(over_sampled_train_x)
        x_train, x_test, y_train, y_test = train_test_split(x, over_sampled_train_y, test_size=.1)
        rfc = RandomForestClassifier(n_estimators=1500, n_jobs=-1, max_depth=15, min_samples_split=5,
                                     min_samples_leaf=3)
        rfc.fit(x_train, y_train)
        pred = rfc.predict(x_test)
        acc = accuracy_score(y_test, pred) * 100
        print('accuracy_score', acc)
        # Classification Report
        print('Classification Report')
        print(classification_report(y_test, pred))
