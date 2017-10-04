import os
import sys
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *
import unittest
import pickle
from math import isclose
import pebble

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'


class RawPValuesTestCase(unittest.TestCase):

    def setUp(self):
        print ('Loading raw pvalues')
        with open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb') as f:
            raw_association_results, raw_most_varying_feature_idx, raw_filt_transcriptIDs = pickle.load(f)

        self.raw_association_results = raw_association_results
        self.raw_most_varying_feature_idx = raw_most_varying_feature_idx
        self.raw_filt_transcriptIDs = raw_filt_transcriptIDs


    def test_raw_pvalues_indexes(self):
        N = 500
        M = 2000
        k = 1

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']



        # with open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb') as f:
        #     corrected_association_results, corrected_most_varying_feature_idx, corrected_filt_transcriptIDs = pickle.load(f)

        feature_assertions = []
        transcript_assertions = []
        for m in MODELS:
            for a in AGGREGATIONS:
                all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', m, a, N)
                filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
                for ps in SIZES:
                    filt_features = all_filt_features[ps]
                    key = '{}_{}_{}_{}'.format('Lung',a,m,ps)
                    print(key)

                    feature_assertion = all(self.raw_most_varying_feature_idx[key] == most_varying_feature_idx)
                    transcript_assertion = all(self.raw_filt_transcriptIDs == filt_transcriptIDs)
                    if not feature_assertion or not transcript_assertion:
                        import pdb; pdb.set_trace()
                    feature_assertions.append(feature_assertion)
                    transcript_assertions.append(transcript_assertion)

        assert all(feature_assertions) and all(transcript_assertions)

    def test_example_raw_pvalues(self):
        N = 500
        M = 2000
        k = 1

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']

        def my_isclose(x, y, abs_tol=1e-3):
            if np.isnan(x) and np.isnan(y):
                return True
            else:
                return isclose(x, y, abs_tol=abs_tol)



        R_assertions = []
        pv_assertions = []
        _, _, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', 'retrained', 'median', N)
        filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
        for m in MODELS:
            for a in AGGREGATIONS:
                all_filt_features, most_varying_feature_idx, _, _, _, _, _, _ = filter_features_across_all_patchsizes('Lung', m, a, N)

                for ps in SIZES:

                    filt_features = all_filt_features[ps]
                    key = '{}_{}_{}_{}'.format('Lung',a,m,ps)
                    print("Checking {}".format(key))

                    # Choosing 100 random transcript / feature pairs.
                    R_assertion = []
                    pv_assertion = []
                    for k in range(100):
                        f = np.random.choice(N)
                        t = np.random.choice(M)
                        computed_associations = self.raw_association_results[key]
                        R, pv = pearsonr(filt_features[:,f], filt_expression[:,t])
                        R_test = my_isclose(computed_associations[0][f][t], R, abs_tol=1e-3)
                        pv_test = my_isclose(computed_associations[1][f][t], pv, abs_tol=1e-5)
                        if not R_test or not pv_test:
                            import pdb; pdb.set_trace()

                        R_assertion.append(R_test)
                        pv_assertion.append(pv_test)

                    R_assertions.append(all(R_assertion))
                    pv_assertions.append(all(pv_assertion))

        assert all(R_assertions) and all(pv_assertions)



class PCCorrectedPValuesTestCase(unittest.TestCase):

    def setUp(self):
        print ('Loading PC corrected pvalues')
        with open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb') as f:
            corrected_association_results, corrected_most_varying_feature_idx, corrected_filt_transcriptIDs = pickle.load(f)

        self.corrected_association_results = corrected_association_results
        self.corrected_most_varying_feature_idx = corrected_most_varying_feature_idx
        self.corrected_filt_transcriptIDs = corrected_filt_transcriptIDs



    def test_corrected_pvalues_indexes(self):
        N = 500
        M = 2000
        k = 1

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        PCs = [1,2,3,4,5]



        # with open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb') as f:
        #     corrected_association_results, corrected_most_varying_feature_idx, corrected_filt_transcriptIDs = pickle.load(f)

        feature_assertions = []
        transcript_assertions = []
        for a in AGGREGATIONS:
            for m in MODELS:
                for pc in PCs:
                    all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', m, a, N, pc_correction=pc)
                    filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
                    for ps in SIZES:

                        filt_features = all_filt_features[ps]
                        key = '{}_{}_{}_{}_{}'.format('Lung', a, m, ps, pc)
                        print("Checking {}".format(key))

                        # computed_associations = self.corrected_association_results[key]
                        feature_assertion = all(self.corrected_most_varying_feature_idx[key] == most_varying_feature_idx)
                        transcript_assertion = all(self.corrected_filt_transcriptIDs == filt_transcriptIDs)
                        if not feature_assertion or not transcript_assertion:
                            import pdb; pdb.set_trace()
                        feature_assertions.append(feature_assertion)
                        transcript_assertions.append(transcript_assertion)

        assert all(feature_assertions) and all(transcript_assertions)


    def test_example_corrected_pvalues(self):
        N = 500
        M = 2000
        k = 1

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        PCs = [1,2,3,4,5]

        def my_isclose(x, y, abs_tol=1e-3):
            if np.isnan(x) and np.isnan(y):
                return True
            else:
                return isclose(x, y, abs_tol=abs_tol)



        R_assertions = []
        pv_assertions = []
        _, _, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', 'retrained', 'median', N)
        filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
        for a in AGGREGATIONS:
            for m in MODELS:
                for pc in PCs:
                # pc = 2
                    all_filt_features, most_varying_feature_idx, _, _, _, _, _, _ = filter_features_across_all_patchsizes('Lung', m, a, N, pc_correction=pc)

                    for ps in SIZES:

                        filt_features = all_filt_features[ps]
                        key = '{}_{}_{}_{}_{}'.format('Lung', a, m, ps, pc)
                        print("Checking {}".format(key))
                        # import pdb; pdb.set_trace()



                        # Choosing 100 random transcript / feature pairs.
                        R_assertion = []
                        pv_assertion = []
                        for k in range(100000):
                            f = np.random.choice(N)
                            t = np.random.choice(M)
                            computed_associations = self.corrected_association_results[key]

                            R, pv = pearsonr(filt_features[:,f], filt_expression[:,t])
                            R_test = my_isclose(computed_associations[0][f][t], R, abs_tol=1e-3)
                            pv_test = my_isclose(computed_associations[1][f][t], pv, abs_tol=1e-3)
                            # import pdb; pdb.set_trace()
                            if not R_test or not pv_test:
                                import pdb; pdb.set_trace()


                            R_assertion.append(R_test)
                            pv_assertion.append(pv_test)


                        R_assertions.append(all(R_assertion))
                        pv_assertions.append(all(pv_assertion))

        assert all(R_assertions) and all(pv_assertions)




class FilterCorrectionTestCase(unittest.TestCase):

    # def setUp(self):
        # print ('Loading TF corrected pvalues')
        # with open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb') as f:
        #     corrected_association_results, corrected_most_varying_feature_idx, corrected_filt_transcriptIDs = pickle.load(f)
        #
        # self.corrected_association_results = corrected_association_results
        # self.corrected_most_varying_feature_idx = corrected_most_varying_feature_idx
        # self.corrected_filt_transcriptIDs = corrected_filt_transcriptIDs

    def filter_and_correct_expression_and_image_features(self):
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        TFs = [1,2,3,4,5]
        N = 500
        M = 2000
        k = 1
        filter_and_correct_expression_and_image_features('Lung', 'retrained', 'median', '256', M, k, pc_correction=5, tf_correction=False)

def fibonacci(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return fibonacci(n - 1) + fibonacci(n - 2)

class GProfilerParallel(unittest.TestCase):
    def test_parallel(self):
        from pebble import ProcessPool
        from concurrent.futures import TimeoutError
        from tqdm import tqdm


        pbar = tqdm(total=50)
        with ProcessPool(max_workers=16) as pool:
            future = pool.map(fibonacci, range(30), timeout=10)
            results = future.result()

            all_results = []
            while True:
                try:
                    result = next(results)
                    all_results.append(result)
                    pbar.update(1)
                except StopIteration:
                    break
                except TimeoutError as error:
                    all_results.append(None)
                    pbar.update(1)
                    print("function took longer than %d seconds" % error.args[1])
                except ProcessExpired as error:
                    all_results.append(None)
                    pbar.update(1)
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except Exception as error:
                    all_results.append(None)
                    print("function raised %s" % error)
                    print(error.traceback)  # Python's traceback of remote process
            print (all_results)
            print (len(all_results))
            import pdb; pdb.set_trace()
            # all_results









if __name__ == '__main__':
    unittest.main()
