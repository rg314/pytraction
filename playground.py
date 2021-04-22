#-----------------------------------------------------------------

def run_it(threaded=False):
    if threaded:
        from pytraction.core_threaded import TractionForce
        import random
    else:
        from pytraction.core import TractionForce
    from pytraction.utils import plot

    pix_per_mu = 1.3 # The number of pixels per micron 
    E = 100 # Youngs modulus in Pa

    img_path = 'data/example1/e01_pos1_axon1.tif'
    ref_path = 'data/example1/e01_pos1_axon1_ref.tif'

    traction_obj = TractionForce(pix_per_mu, E=E)
    img, ref, _ = traction_obj.load_data(img_path, ref_path)
    log = traction_obj.process_stack(img, ref)

    plot(log, frame=0)
    # plt.show()

#-----------------------------------------------------------------

if __name__ == "__main__":
    import time

    runs = {"non-threaded":[], "threaded":[]}
    for x in range(10):
        time_start = time.time_ns()
        run_it()
        time_end = time.time_ns()
        time_taken_ns = time_end - time_start
        print("runtime non-threaded (ns):", time_taken_ns)
        runs['non-threaded'].append((time_taken_ns, None))

        time_start = time.time_ns()
        num_workers = run_it(threaded=True)
        time_end = time.time_ns()
        time_taken_ns = time_end - time_start
        print("runtime threaded (ns):", time_taken_ns)
        runs['threaded'].append((time_taken_ns, None))

    print(runs)

# TODO: test that both non-threaded and threaded actually return the same thing (i.e. that it still works when multiprocessing)
