import io
import pickle
from pathlib import Path
import numpy as np
import pytest
from app.engine import MODEL_IDS, rank_candidates
from ml.artifacts import PrimitiveUnpickler, read_bundle, write_bundle
from ml.prepare_wider import annotations, select
from ml.evaluate_lfw import calibrate, metrics


def test_executable_pickle_is_rejected():
    payload=pickle.dumps(Path('never-executed'))
    with pytest.raises(pickle.UnpicklingError):
        PrimitiveUnpickler(io.BytesIO(payload)).load()


def test_deployed_manifest_checksums():
    bundle=read_bundle(Path(__file__).parents[1]/'models/pipeline.pkl')
    assert bundle['recognizer']['dimensions']==128
    assert bundle['deployment']['disaster_validated'] is False


def test_annotation_zero_faces_and_stratification():
    rows=list(annotations('event/a.jpg\n0\n0 0 0 0 0 0 0 0 0 0\nevent/b.jpg\n1\n1 2 30 40 0 0 0 0 0 0\n'))
    assert rows[0][1]==[] and rows[1][1][0][:4]==[1,2,30,40]
    values=[('a/'+str(i),[]) for i in range(20)]+[('b/'+str(i),[]) for i in range(20)]
    sample=select(values,8,42)
    assert sample==select(values,8,42)
    assert sum(x[0].startswith('a/') for x in sample)==4


def test_ranking_ignores_ear_and_rejects_invalid_face_vectors():
    q=np.array([1.,0.],dtype=np.float32)
    refs=[{'modality':'ear','embedding':[1.,0.],'model':MODEL_IDS['ear'],'person_id':'ear-only'},
          {'modality':'face','embedding':[2.,0.],'model':MODEL_IDS['face'],'person_id':'face'}]
    out=rank_candidates({'face':q,'ear':q},refs,{'face':.8})
    assert len(out)==1 and out[0]['person_id']=='face' and out[0]['components']=={'face':1.0}
    assert out[0]['score_is_probability'] is False and out[0]['status']=='pending'
    assert rank_candidates({'ear':q},refs,{'face':.8})==[]
    refs[1]['embedding']=[float('nan'),0.]
    with pytest.raises(ValueError):rank_candidates({'face':q},refs,{'face':.8})


def test_calibration_uses_genuine_and_impostor_pairs():
    scores=[{'score':.2+i/1000,'same':False} for i in range(120)]+[{'score':.7+i/1000,'same':True} for i in range(120)]
    threshold=calibrate(scores,.01)
    result=metrics(scores,threshold)
    assert result['true_accept_rate']==1 and result['false_accept_rate']<=.01
    assert result['far_wilson_95_upper']>result['false_accept_rate']
    with pytest.raises(ValueError):calibrate(scores[:50],.01)
