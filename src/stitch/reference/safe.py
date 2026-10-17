"""Source-group exclusion around preserved reference statistics."""
from ._hgb_math import reference_features
from ._cnf_math import spatial_features


def validate_disjoint(reference, query):
    for column in ['tic_id', 'physical_group_id']:
        if column not in reference or column not in query:
            raise ValueError(f'Reference and query require {column}')
        if reference[column].isna().any() or query[column].isna().any():
            raise ValueError(f'Missing source identity: {column}')
        if set(reference[column].astype(str)) & set(query[column].astype(str)):
            raise ValueError(f'Reference bank contains a query {column}')
    if 'role' in reference and not reference.role.eq('train').all():
        raise ValueError('Only training labels may enter reference features')
    if not reference.systematic_score.gt(.95).all():
        raise ValueError('Reference sectors must individually pass TARS >0.95')


def hgb_features(reference, query):
    validate_disjoint(reference, query)
    return reference_features(reference, query)


def cnf_features(reference, query):
    validate_disjoint(reference, query)
    return spatial_features(reference, query)
