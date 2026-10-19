from slm_lab.codenet_description_v1 import parse_description


def test_output_for_sample_input_is_an_output_and_preserves_pre_spacing():
    row = parse_description('<h2>Sample Input 1</h2><pre>\n 1  2\n3\n</pre>'
                            '<h2>Output for the Sample Input 1</h2><pre>\n 6\n</pre>')
    assert row['samples'] == [{'number': '1', 'input': ' 1  2\n3', 'output': ' 6'}]
    assert row['recognized_samples_usable']


def test_format_blocks_are_not_guessed_to_be_test_inputs():
    row = parse_description('<h2>Input</h2><pre>n a_1 ... a_n</pre>'
                            '<h2>Output</h2><pre>answer</pre>')
    assert not row['samples'] and not row['recognized_samples_usable']
    assert 'n a_1' in row['text']


def test_identical_bilingual_samples_deduplicate_without_losing_text():
    row = parse_description('<h3>入力例 1</h3><pre>1</pre><h3>出力例 1</h3><pre>2</pre>'
                            '<h3>Sample Input 1</h3><pre>1</pre><h3>Sample Output 1</h3><pre>2</pre>')
    assert row['samples'] == [{'number': '1', 'input': '1', 'output': '2'}]
    assert row['recognized_samples_usable'] and '入力例' in row['text']


def test_conflicting_or_mismatched_samples_are_not_certified_usable():
    row = parse_description('<h2>Sample Input 1</h2><pre>1</pre>'
                            '<h2>Sample Input 1</h2><pre>3</pre>'
                            '<h2>Sample Output 1</h2><pre>2</pre>')
    assert row['sample_parse_issues'] and not row['recognized_samples_usable']
    row = parse_description('<h2>Sample Input 1</h2><pre>1</pre>'
                            '<h2>Sample Output 2</h2><pre>2</pre>')
    assert not row['samples'] and len(row['sample_parse_issues']) == 2


def test_math_structure_survives_html_export_and_executable_scripts_are_omitted():
    row = parse_description('<p>a<sub>i</sub> &le; n<sup>2</sup></p>'
                            '<script type="math/tex">x^2 + y^2</script>'
                            '<script>alert("omit");</script>')
    assert 'a_{i} ≤ n^{2}' in row['text'] and '$x^2 + y^2$' in row['text']
    assert 'alert' not in row['text']


def test_unavailable_media_is_flagged_even_with_alt_text():
    row = parse_description('<p>Use this graph.</p><img src="graph.png" alt="graph"><svg></svg>')
    assert row['requires_media_review'] and row['image_references'][0]['src'] == 'graph.png'


def test_unclosed_example_does_not_create_a_complete_oracle():
    row = parse_description('<h2>Sample Input</h2><pre>1')
    assert row['sample_parse_issues'] and not row['recognized_samples_usable']


def test_full_width_sample_numbers_match_but_sample_characters_are_preserved():
    row = parse_description('<h2>入力例１</h2><pre>１</pre>'
                            '<h2>出力例１</h2><pre>２</pre>'
                            '<h2>入力例２</h2><pre>３</pre>'
                            '<h2>出力例２</h2><pre>４</pre>')
    assert row['samples'] == [{'number': '1', 'input': '１', 'output': '２'},
                              {'number': '2', 'input': '３', 'output': '４'}]
    assert row['recognized_samples_usable']


def test_unclosed_suppressed_block_is_not_silent_complete_text():
    row = parse_description('<h2>Sample Input</h2><pre>1</pre>'
                            '<h2>Sample Output</h2><pre>2</pre><style>unclosed')
    assert row['samples'] and row['sample_parse_issues']
    assert not row['recognized_samples_usable']


def test_explanatory_matrix_after_output_is_context_not_a_second_sample_output():
    row = parse_description('<h2>Sample Input 1</h2><pre>3</pre>'
                            '<h2>Sample Output 1</h2><pre>16</pre>'
                            '<p>The following submatrix gives this sum.</p><pre>-4 5\n7 8</pre>')
    assert row['samples'] == [{'number': '1', 'input': '3', 'output': '16'}]
    assert row['recognized_samples_usable'] and '-4 5\n7 8' in row['text']


def test_consecutive_pre_blocks_without_explanation_remain_ambiguous():
    row = parse_description('<h2>Sample Input 1</h2><pre>3</pre>'
                            '<h2>Sample Output 1</h2><pre>16</pre><pre>17</pre>')
    assert row['sample_parse_issues'] and not row['recognized_samples_usable']
