export const initialState = {
  loading: true,
  data: [],
  error: null,
};

export const actionTypes = {
  FETCH_DATA_START: 'FETCH_DATA_START',
  FETCH_DATA_SUCCESS: 'FETCH_DATA_SUCCESS',
  FETCH_DATA_ERROR: 'FETCH_DATA_ERROR',
  SET_FILTERED_DATA: 'SET_FILTERED_DATA',
};

export const reducer = (state, action) => {
  switch (action.type) {
    case actionTypes.FETCH_DATA_START:
      return {
        ...state,
        loading: true,
        error: null,
      };
    case actionTypes.FETCH_DATA_SUCCESS:
      return {
        ...state,
        loading: false,
        data: action.payload,
      };
    case actionTypes.FETCH_DATA_ERROR:
      return {
        ...state,
        loading: false,
        error: action.payload,
      };
    case actionTypes.SET_FILTERED_DATA:
      return {
        ...state,
        filteredData: action.payload,
      };      
    default:
      return state;
  }
};
