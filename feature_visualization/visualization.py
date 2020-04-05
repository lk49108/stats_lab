import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np

class FeaturePlot:

    def __init__(self, extracted_features, feature_generator, signal_treat=['glu', 'eth', 'sal', 'nea']):
        self.data = extracted_features
        self.feature_generator = feature_generator
        self.chunk_duration = self.feature_generator.chunk_duration
        self.treatments = signal_treat
        self.mouse_ids = list(self.feature_generator.mouse_ids)
        self.list_indexes_mouse_treatment = []
        self.list_indexes_treatment = []
        self.indexIdentifier()
        self.dimension = None
        #self.position = None

    def plotFeatures(self, plot_type = ['subjectLineplot','overallLineplot','boxplot','histogram']):
        dispatcher = {'subjectLineplot': self.subjectLineplot,
                      'overallLineplot': self.overallLineplot,
                      'boxplot': self.boxplot,
                      'histogram': self.histogram
                      }
        if plot_type == 'all':
            plot_type = ['subjectLineplot', 'overallLineplot', 'boxplot', 'histogram']
        if isinstance(plot_type,list):
            pass
        else:
            plot_type = [plot_type]
        self.plot_type = plot_type
        self.subplotposition()

        def call_func(func):
            return (dispatcher[func]())
        for func in plot_type:
            call_func(func)

    def indexIdentifier(self):
        mouse_ids = self.mouse_ids
        indices = self.data.index
        for j in mouse_ids:
            matched_indexes_mouse = [row_id for row_id in indices if str(j) in row_id]
            for treat in self.treatments:
                matched_indexes_mouse_treatment = [row_id for row_id in matched_indexes_mouse if str(j) + '_' + treat in row_id]
                self.list_indexes_mouse_treatment.append(matched_indexes_mouse_treatment)
        for treat in self.treatments:
            matched_indexes_treatment = [row_id for row_id in indices if treat in row_id]
            self.list_indexes_treatment.append(matched_indexes_treatment)

    def subjectLineplot(self):
        feature_number = len(self.data.columns)
        mouse_ids = self.mouse_ids
        chunk_duration = self.chunk_duration
        for f in range(feature_number):
            fig = plt.figure(num=f + 1, figsize=(12, 12))
            i, j = 0, 4
            for mouse in mouse_ids:
                ax = fig.add_subplot(int(str(self.dimension)+str(self.position_subject + mouse_ids.index(mouse))))
                for values in self.list_indexes_mouse_treatment[i:j]:
                    ax.plot(list(range(len(values))), self.data.loc[values, self.data.columns[f]], label=values[1][-3:])
                plt.xlabel("Data chunk (" + str(chunk_duration) + ' Min.)')
                plt.ylabel("Feature value")
                plt.title('Mouse ' + str(mouse))
                i, j = i + 4, j + 4
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            fig.suptitle(str(self.data.columns[f]), x= 0.5, y = 1, fontsize=16)
            fig.tight_layout(pad=3.0)


    def overallLineplot(self):
        feature_number = len(self.data.columns)
        mouse_ids = self.mouse_ids
        chunk_duration = self.chunk_duration
        for f in range(feature_number):
            fig = plt.figure(num=f + 1, figsize=(12, 12))
            ax = fig.add_subplot(int(str(self.dimension)+str(self.position_over)))
            for values in self.list_indexes_mouse_treatment:
                ax.plot(list(range(len(values))), self.data.loc[values, self.data.columns[f]], label=values[1][-3:])
            plt.xlabel("Data chunk (" + str(chunk_duration) + ' Min.)')
            plt.ylabel("Feature value")
            plt.title('Mouse ' + str(mouse_ids))
            plt.legend(self.treatments, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            fig.suptitle(str(self.data.columns[f]), x= 0.5, y = 1, fontsize=16, horizontalalignment='center',
                         verticalalignment='top')
            fig.tight_layout(pad=3.0)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] * len(mouse_ids)
            for i, j in enumerate(ax.lines):
                j.set_color(colors[i])

    def boxplot(self):
        feature_number = len(self.data.columns)
        mouse_ids = self.mouse_ids
        chunk_duration = self.chunk_duration
        for f in range(feature_number):
            fig = plt.figure(num=f + 1, figsize=(12, 12))
            ax = fig.add_subplot(int(str(self.dimension)+str(self.position_box)))
            ax.boxplot([self.data.loc[self.list_indexes_treatment[0], self.data.columns[f]],
                        self.data.loc[self.list_indexes_treatment[1], self.data.columns[f]],
                        self.data.loc[self.list_indexes_treatment[2], self.data.columns[f]],
                        self.data.loc[self.list_indexes_treatment[3], self.data.columns[f]]],
                       notch = True, labels = self.treatments, showmeans = True)
            plt.xlabel("Treatments (Chunk duration " + str(chunk_duration) + ' Min.)')
            plt.ylabel("Feature value")
            plt.title('Mouse ' + str(mouse_ids))
            fig.suptitle(str(self.data.columns[f]), x= 0.5, y = 1, fontsize=16, horizontalalignment='center',
                         verticalalignment='top')
            fig.tight_layout(pad=3.0)

    def histogram(self):
        feature_number = len(self.data.columns)
        mouse_ids = self.mouse_ids
        chunk_duration = self.chunk_duration
        for f in range(feature_number):
            fig = plt.figure(num=f + 1, figsize=(12, 12))
            for i in range(len(self.treatments)):
                ax = fig.add_subplot(int(str(self.dimension)+str(self.position_hist + i)))
                ax.hist(self.data.loc[self.list_indexes_treatment[i], self.data.columns[f]],
                        bins=len(self.list_indexes_treatment[i]), rwidth=0.95)
                #ax.hist(self.data.loc[self.list_indexes_treatment[i], self.data.columns[f]],
                #        bins = sorted(self.data.loc[self.list_indexes_treatment[i], self.data.columns[f]]), rwidth = 0.9)
                plt.xlabel("Feature value (Chunk" + str(chunk_duration) + ' Min.)')
                plt.ylabel("Occurences")
                plt.title('Mouse ' + str(mouse_ids) +' Treatment '+ self.treatments[i])
            fig.suptitle(str(self.data.columns[f]), x= 0.5, y = 1, fontsize=16, horizontalalignment='center',
                         verticalalignment='top')
            fig.tight_layout(pad=3.0)

    def subplotposition(self):
        dimension, position_over, position_box, position_subject, position_hist = 0, 0, 0, 0, 0
        if len(self.mouse_ids) > 4:
            raise ValueError('Too many subjects for autodetection!')
        elif len(self.plot_type) == 1:
            if 'histogram' not in self.plot_type and 'subjectLineplot' not in self.plot_type:
                dimension, position_over, position_box = 11, 1, 1
            else:
                dimension, position_hist, position_subject = 22, 1, 1
        elif len(self.plot_type) == 2 :
            if 'histogram' not in self.plot_type and 'subjectLineplot' not in self.plot_type:
                dimension, position_over, position_box  = 21, 1, 2
            elif 'histogram' in self.plot_type and 'subjectLineplot' in self.plot_type:
                if 0 < len(self.mouse_ids) < 3:
                    dimension, position_subject, position_hist  = 32, 1, 3
                elif 2 < len(self.mouse_ids) < 5:
                    dimension, position_subject, position_hist  = 33, 1, 4
            elif 'subjectLineplot' in self.plot_type:
                if 1 < len(self.mouse_ids) < 4:
                    dimension, position_subject, position_over, position_box  = 22, 1, 4, 4
                elif len(self.mouse_ids) == 4:
                    dimension, position_subject, position_over, position_box  = 32, 1, 5, 5
                else:
                    dimension, position_subject, position_over, position_box = 21, 1, 2, 2
            elif 'histogram' in self.plot_type:
                dimension, position_hist, position_over, position_box = 32, 1, 5, 5
        elif len(self.plot_type) == 3 :
            if 'histogram' in self.plot_type and 'subjectLineplot' not in self.plot_type:
                dimension, position_hist, position_over, position_box = 32, 1, 5, 6
            elif 'histogram' in self.plot_type and 'subjectLineplot' in self.plot_type:
                if len(self.mouse_ids) == 1:
                    dimension, position_subject, position_over, position_box, position_hist = 32, 1, 2, 2, 5
                elif 1 < len(self.mouse_ids) < 4:
                    dimension, position_subject, position_over, position_box, position_hist = 33, 1, 4, 4, 5
                elif len(self.mouse_ids) == 4:
                    dimension, position_subject, position_over, position_box, position_hist = 33, 1, 5, 5, 6
            elif 'subjectLineplot' in self.plot_type:
                if 0 < len(self.mouse_ids) < 3:
                    dimension, position_subject, position_over, position_box = 22, 1, 3, 4
                if  2 < len(self.mouse_ids) < 5:
                    dimension, position_subject, position_over, position_box  = 32, 1, 5, 6
        elif len(self.plot_type) == 4 :
            if 0 < len(self.mouse_ids) < 4:
                dimension, position_subject, position_over, position_box, position_hist = 33, 1, 4, 5, 6
            elif len(self.mouse_ids) == 4:
                dimension, position_subject, position_over, position_box, position_hist = 43, 1, 5, 6, 7
        self.dimension = dimension
        self.position_over = position_over
        self.position_box = position_box
        self.position_subject = position_subject
        self.position_hist = position_hist

'''        for ff in range(feature_number):
            fig = plt.figure(num=ff + 1, figsize=(12, 12))
            axe = fig.add_subplot(2, 2, 3)
            
            axe.plot(list(range(len(list_indexes[0]))), mouse.iloc[list_indexes[0], ff],
                     list(range(len(list_indexes[1]))),
                     mouse.iloc[list_indexes[1], ff], list(range(len(list_indexes[2]))),
                     mouse.iloc[list_indexes[2], ff],
                     list(range(len(list_indexes[3]))), mouse.iloc[list_indexes[3], ff], color='red')

            plt.title('Mice ' + str(mouse_ids))
            plt.xlabel("Data chunk (" + str(chunk_duration) + ' Min.)')
            plt.ylabel("Feature value")
            plt.legend(treatments, loc='upper right')  # [a,b,c,d],treatments , loc='upper right
            '''

'''previous indexIdentifiers
            #             length = len(indices)
        #             matched_indexes_mouse = []
        # while i < length:
        #matched_indexes_mouse_treatment = []
            #     row_id = indices[i]
            #     if str(j) in row_id:
            #         matched_indexes_mouse.append(row_id)
            #     i += 1

            #mouse = feature_data.iloc[matched_indexes, :]
            #indices = mouse.index
            #length = len(indices)
        #         length = len(matched_indexes_mouse)
        #         list_indexes_mouse_treatment = []
        #         for treat in self.treatments:
        #             i = 0
        #             matched_indexes_mouse_treatment = []
        #             while i < length:
        #                 row_id = matched_indexes_mouse[i]
        #                 if str(j) + '_' + treat in row_id:
        #                     matched_indexes_mouse_treatment.append(row_id)
        #                 i += 1
        #             list_indexes_mouse_treatment.append(matched_indexes_mouse_treatment)
        # self.list_iexes_mouse_treatment = list_indexes_mouse_treatment.append(list_indexes_mouse_treatment)'''


'''    def plotFeatures(self, extracted_features, signal_treat = ['glu', 'eth', 'sal', 'nea']):
        #feature_data = featureDataPreparation(mouse, signal_type, brain_half, mouse_ids, treatments)
        self.treatments = signal_treat
        self.extracted_features = extracted_features
        feature_list, feature_data, mouse_ids, treatments, chunk_duration = feature_data
        print(feature_data)
        feature_number = len(feature_data.columns)
        subplot_position = 1
        for j in mouse_ids:
            i = 0
            indices = feature_data.index
            length = len(indices)
            matched_indexes = []
            while i < length:
                if str(j) in indices[i]:
                    matched_indexes.append(i)
                i += 1

            mouse = feature_data.iloc[matched_indexes,:]
            indices = mouse.index
            length = len(indices)
            list_indexes = []
            for k in range(len(treatments)):
                i = 0
                matched_indexes = []
                while i < length:
                    if str(j)+'.'+str(k+1) in indices[i]:
                        matched_indexes.append(i)
                    i += 1
                list_indexes.append(matched_indexes)
                for f in range(feature_number):
                        fig = plt.figure(num = f+1, figsize=(12,12))
                        ax = fig.add_subplot(2, 2,subplot_position)
                        ax.plot(list(range(len(matched_indexes))), mouse.iloc[matched_indexes,f], label = str(treatments[k]))
                        plt.title('Mouse '+str(j))
                        plt.xlabel("Data chunk ("+str(chunk_duration)+' Min.)')
                        plt.ylabel("Feature value")
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        fig.suptitle(str(feature_data.columns[f]), fontsize=16, horizontalalignment='center', verticalalignment='top')
                        fig.tight_layout(pad=3.0)
            subplot_position += 1
            for ff in range(feature_number):
                fig = plt.figure(num=ff + 1, figsize=(12,12))
                axe = fig.add_subplot(2, 2, 3)
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']*len(mouse_ids)
                axe.plot(list(range(len(list_indexes[0]))), mouse.iloc[list_indexes[0],ff], list(range(len(list_indexes[1]))),
                         mouse.iloc[list_indexes[1],ff], list(range(len(list_indexes[2]))), mouse.iloc[list_indexes[2],ff],
                         list(range(len(list_indexes[3]))), mouse.iloc[list_indexes[3],ff], color = 'red')
                for i, j in enumerate(axe.lines):
                    j.set_color(colors[i])
                plt.title('Mice ' + str(mouse_ids))
                plt.xlabel("Data chunk ("+str(chunk_duration)+' Min.)')
                plt.ylabel("Feature value")
                plt.legend(treatments , loc='upper right') #[a,b,c,d],treatments , loc='upper right'
        plt.show()'''