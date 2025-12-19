# cython: language_level=3

cdef create_fastrlock()

# acquire the lock of a FastRlock instance
# 'current_thread' may be -1 for the current thread
cdef bint lock_fastrlock(rlock, long current_thread, bint blocking) except -1

# release the lock of a FastRlock instance
cdef int unlock_fastrlock(rlock) except -1
