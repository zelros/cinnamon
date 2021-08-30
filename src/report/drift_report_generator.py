from .i_report_generator import IReportGenerator
from ..drift.i_drift_explainer import IDriftExplainer

import pickle as pkl
from pweave import weave
import pkgutil


class DriftReportGenerator(IReportGenerator):

    @staticmethod
    def generate(drif_explainer: IDriftExplainer, output_path: str, min_cat_weight):
        with open('report_data.pkl', 'wb') as f:
            pkl.dump({'drift_explainer': drif_explainer, 'min_cat_weight': min_cat_weight}, f)
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