from .i_report_generator import IReportGenerator

import pickle as pkl
from pweave import weave
import pkgutil


class DriftReportGenerator(IReportGenerator):

    @staticmethod
    def generate(drif_explainer, output_path: str, max_n_cat):
        with open('report_data.pkl', 'wb') as f:
            pkl.dump({'drift_explainer': drif_explainer, 'max_n_cat': max_n_cat}, f)
        data = pkgutil.get_data(__name__, '/drift_report_template.pmd')
        import tempfile
        with tempfile.NamedTemporaryFile('w') as fp:
            fp.write(data.decode('utf-8'))
            weave(fp.name, informat='markdown', # on part de l'endroit où le code est exécuter donc dans le notebook
                  output=output_path)

        #with open('template.pmd', 'w') as f:
        #    f.write(data.decode('utf-8'))
        #print(data)
        #print(type(data))
        #print(data.decode('utf-8'))
        #print(type(data.decode('utf-8')))
        weave('template.pmd',  # on part de l'endroit où le code est exécuter donc dans le notebook
              output=output_path)

        #weave('src/report/drift_report_template.pmd', # on part de l'endroit où le code est exécuter donc dans le notebook
        #      output=path)
