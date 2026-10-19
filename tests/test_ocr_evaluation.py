from slm_lab.ocr_evaluation import transcription_scores,select


def test_transcription_errors_penalize_missing_and_hallucinated_text():
    assert transcription_scores('Café\n123','Cafe\u0301   123')['exact_match']
    case=transcription_scores('STOP','stop')
    assert not case['exact_match'] and case['case_insensitive_exact_match']
    deletion=transcription_scores('turn left','turn')
    assert deletion['word_edits']==1 and deletion['character_edits']==5
    insertion=transcription_scores('a','abcdef')
    assert insertion['character_edits']/insertion['reference_characters']>1


def test_ocr_sample_does_not_let_many_crops_of_one_photo_dominate_a_task():
    rows=[{'id':str(i),'group_id':'same-photo','task':'word_ocr'} for i in range(30)]
    rows += [{'id':'another','group_id':'another-photo','task':'word_ocr'},
             {'id':'document','group_id':'document','task':'document_ocr'}]
    chosen=select(rows,10)
    assert len(chosen)==3
    assert sum(r['group_id']=='same-photo' for r in chosen)==1
