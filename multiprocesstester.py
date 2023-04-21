import time, concurrent.futures
from multiprocessing import Pool, freeze_support




start = time.perf_counter()



def do_something(timeval):
    timeval = int(timeval)
    print(f"Sleeping for {timeval} second")
    time.sleep(timeval)
    return f"Done Sleeping for {timeval} seconds."

sleeptimes = [5,4,3,2,1]
# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         f = executor.map(do_something, sleeptimes)

#         for result in f:
#             print(result)

#     finish = time.perf_counter()

#     print(f"Finished in {finish-start:.3f} second(s)")


def pleasesleep(sleeptimes=sleeptimes):
    with Pool() as pool:
        f = pool.map(do_something, sleeptimes)

        for result in f:
            print(result)

if __name__ == '__main__':
    freeze_support()

    pleasesleep(sleeptimes=sleeptimes)


    finish = time.perf_counter()

    print(f"Finished in {finish-start:.3f} second(s)")