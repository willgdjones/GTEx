import os
import sys
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *
import unittest
import pickle
from math import isclose

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



class CorrectedPValuesTestCase(unittest.TestCase):

    def setUp(self):
        print ('Loading corrected pvalues')
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
        N = 50
        M = 200
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
                        for k in range(100):
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


class ComputePvaluesTestCase(unittest.TestCase):


    def test_compute_raw_pvalues(self):
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATION = 'mean'
        MODEL = 'raw'
        N = 5
        M = 10
        k = 1

        all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', MODEL, AGGREGATION, N)
        filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
        filt_features = all_filt_features[128]

        res = compute_pearsonR(filt_features, filt_expression)

        R_assertions = []
        pv_assertions = []
        for i in range(10):
            f = np.random.choice(N)
            t = np.random.choice(M)
            R, pv = pearsonr(filt_features[:,f], filt_expression[:,t])
            R_assertion = res[0][f][t] == R
            pv_assertion = res[1][f][t] == pv
            R_assertions.append(R_assertion)
            pv_assertions.append(pv_assertion)

        assert all(R_assertions) and all(pv_assertions)





        import pdb; pdb.set_trace()

    def test_compute_corrected_pvalues(self):
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATION = 'median'
        MODEL = 'retrained'
        PC = 2
        N = 5
        M = 10
        k = 1

        all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', MODEL, AGGREGATION, N, pc_correction=PC)
        filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
        filt_features = all_filt_features[128]

        res = compute_pearsonR(filt_features, filt_expression)

        R_assertions = []
        pv_assertions = []
        for i in range(10):
            f = np.random.choice(N)
            t = np.random.choice(M)
            R, pv = pearsonr(filt_features[:,f], filt_expression[:,t])
            R_assertion = res[0][f][t] == R
            pv_assertion = res[1][f][t] == pv
            R_assertions.append(R_assertion)
            pv_assertions.append(pv_assertion)

        assert all(R_assertions) and all(pv_assertions)










if __name__ == '__main__':
    unittest.main()
