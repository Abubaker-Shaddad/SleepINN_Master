"""Fixed abbreviation and terminology dictionaries used in retrieval."""

ABBREVIATIONS = {'RLS': 'Restless Legs Syndrome', 'RBD': 'REM Sleep Behavior Disorder', 'OSA': 'Obstructive Sleep Apnea', 'CSA': 'Central Sleep Apnea', 'PLMD': 'Periodic Limb Movement Disorder', 'PLMS': 'Periodic Limb Movements of Sleep', 'PSG': 'Polysomnography', 'MSLT': 'Multiple Sleep Latency Test', 'ICSD': 'International Classification of Sleep Disorders'}

CLINICAL_SYNONYMS = {'rbd': ['rem sleep behavior disorder', 'rem sleep without atonia'], 'rem sleep behavior disorder': ['rbd', 'rswa'], 'narcolepsy': ['hypersomnolence', 'sleep-onset rem periods'], 'osa': ['obstructive sleep apnea', 'sleep-disordered breathing'], 'obstructive sleep apnea': ['osa', 'apnea hypopnea index'], 'insomnia': ['difficulty initiating sleep', 'difficulty maintaining sleep'], 'plmd': ['periodic limb movement disorder', 'periodic limb movements of sleep']}
