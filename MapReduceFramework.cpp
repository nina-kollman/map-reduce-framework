#include "MapReduceFramework.h"
#include "Barrier.h"
#include "pthread.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <cmath>


#define PTHREAD_ERROR "system error: pthread_create function failed"
#define BAD_ALLOC_ERROR "system error: bad allocation"
#define PTHREAD_JOIN_ERROR "system error: pthread_join function failed"


typedef std::mutex mutex;
typedef std::atomic<int> atomic;
struct ThreadContext;
int job_index = 0;
int64_t pow33 = 8589934591;
int64_t reset = 3;


//This is the JobHandle information
typedef struct {
    int id;
    int num_of_threads;
    JobState *state;
    std::atomic<int64_t> *state_counter;
    const InputVec *inputVec;
    atomic *input_vec_index;
    std::vector<IntermediateVec *> *shuffleVec;
    atomic *shuffle_vec_size;
    OutputVec *outputVec;
    atomic *outputvec_size;
    atomic *intermediary_num_of_elements;
    pthread_t *threads;
    ThreadContext *contexts;
    Barrier *barrier;
    const MapReduceClient *client;
    mutex protect_output;
    mutex protect_shuffle_vector;
    mutex protect_state;
    mutex protect_percentage;
    mutex protect_join;
    bool wait_flag;
} JobDetails;

struct ThreadContext {
    int threadID;
    JobDetails *job;
    IntermediateVec *intermediate_vec;
};


/**
 * Updates the percentage of the current job
 * @param job the given job
 */
void update_percentage(void *job)
{
    auto *current_job = (JobDetails *) job;
    auto counter = current_job->state_counter;
    current_job->protect_percentage.lock();
    int64_t cnt = (int64_t)counter->load();
    double num = (double)((cnt & pow33) >> 2) / (double)(cnt >> 33);
    current_job->state->percentage = (float)(num * 100);
    int stage_num = (int)(cnt & reset);
    current_job->state->stage = static_cast<stage_t>(stage_num);
    current_job->protect_percentage.unlock();
}

/**
 * Comparator for sorting Intermediate vector with keys of type K2
 * @param key1 given first key to compare
 * @param key2 given second key to compare
 * @return true when first is bigger than second, else false
 */
bool comparator (const IntermediatePair &key1, const IntermediatePair &key2)
{
    auto first_key = key1.first;
    auto second_key = key2.first;
    return *first_key < *second_key;
}

/**
 * Verifies if two given K2 keys are equal
 * @param key1 first given key
 * @param key2 second given key
 * @return true if key1 == key2, else false
 */
bool is_equal (const K2 *key1, const K2 *key2)
{
    return !(*key1 < *key2) and !(*key2 < *key1);
}

/**
 * Search for a matching vector with the given pair.key type
 * @param job the given current job to search the vector in
 * @param key the given key to search for
 * @return if exist, the index of the matching vector. otherwise -1
 */
int find_vector (void *job, const K2 *key)
{
    auto *current_job = (JobDetails *) job;
    if (current_job->shuffleVec == nullptr)
    {
        current_job->shuffleVec = new (std::nothrow) std::vector<IntermediateVec *> ();
        if (current_job->shuffleVec == nullptr)
        {
            std::cout << BAD_ALLOC_ERROR << std::endl;
            exit(1);
        }

        return -1;
    }
    auto outer_vec = *(current_job->shuffleVec);
    for (int i = 0; i < (int)current_job->shuffleVec->size (); i++)
    {
        const K2 *key1 = (*outer_vec[i])[0].first;
        if (is_equal (key1, key))
        {
            return i;
        }
    }

    return -1;
}

/**
 * Performs the shuffle operation of thread 0
 * @param arg - the context of thread 0
 */
void shuffle (void *arg)
{
    auto *current_context = (ThreadContext *) arg;
    auto *current_job = current_context->job;

    for (int i = 0; i < current_job->num_of_threads; i++)
    {
        K2 *last_key;
        while (!current_job->contexts[i].intermediate_vec->empty ())
        {
            IntermediateVec *vec = nullptr;
            IntermediatePair last_pair = current_job->contexts[i].intermediate_vec->back ();
            auto current_key = last_pair.first;
            int vec_index = find_vector (current_job, current_key);
            if (vec_index == -1)
            {
                vec = new (std::nothrow) IntermediateVec ();
                if (vec == nullptr)
                {
                    std::cout << BAD_ALLOC_ERROR << std::endl;
                    exit(1);

                }
                vec_index = current_job->shuffleVec->size ();
                current_job->shuffleVec->push_back (vec);
                current_job->shuffle_vec_size->fetch_add(1);
            }
            // add pair to shuffle vector and delete it from intermediate vec
            (*current_job->shuffleVec)[vec_index]->push_back (last_pair);
            current_job->state_counter->fetch_add(4);
            current_job->contexts[i].intermediate_vec->pop_back ();
            last_key = current_key;

            //Moving equal keys to the found vector (vector index position)
            while (!current_job->contexts[i].intermediate_vec->empty ())
            {
                last_pair = current_job->contexts[i].intermediate_vec->back ();
                current_key = last_pair.first;
                if (is_equal (current_key, last_key))
                {
                    (*current_job->shuffleVec)[vec_index]->push_back (last_pair);
                    current_job->state_counter->fetch_add(4);
                    current_job->contexts[i].intermediate_vec->pop_back ();
                    last_key = current_key;
                }
                else
                {
                    // Go back to find the relevant vector
                    break;
                }
            }
        }
    }

}

/**
 * This function performs the map phase for each thread
 * @param arg the context of the specific thread
 */
void map_phase (void *arg)
{
    auto *current_context = (ThreadContext *) arg;
    auto *current_job = current_context->job;

    current_job->protect_state.lock ();
    if ((current_job->state_counter->load() & 3) == 0)
    {
        current_job->state_counter->fetch_add(1);
    }
    current_job->protect_state.unlock ();

    int input_vec_size = (int)current_job->inputVec->size ();
    int my_index = current_job->input_vec_index->fetch_add (1);
    while (my_index < input_vec_size)
    {
        auto vec = *(current_job->inputVec);
        auto k1 = vec[my_index].first;
        auto v1 = vec[my_index].second;

        current_job->client->map (k1, v1, current_context);
        current_job->state_counter->fetch_add(4);
        my_index = current_job->input_vec_index->fetch_add (1);
    }
}

/**
 * This function performs the barrier and shuffle phase for each thread
 * @param arg the context of the specific thread
 */
void shuffle_and_barrier_phase (void *arg)
{
    auto *current_context = (ThreadContext *) arg;
    auto *current_job = current_context->job;
    current_job->barrier->barrier ();
    if (!current_context->threadID) // If threadId == 0
        {
        int64_t update_num_of_elements = ((int64_t)current_job->intermediary_num_of_elements->load()
                << 33) + 2;
        current_job->state_counter->exchange(update_num_of_elements);
        shuffle (current_context);
        update_num_of_elements = ((int64_t)current_job->intermediary_num_of_elements->load()
                << 33)+3;
        current_job->state_counter->exchange(update_num_of_elements);
        current_job->barrier->barrier ();
        }
    else
        {
            current_job->barrier->barrier ();
        }

}

/**
 * This function performs the reduce phase for each thread
 * @param arg the context of the specific thread
 */
void reduce_phase (void *arg)
{
    auto *current_context = (ThreadContext *) arg;
    auto *current_job = current_context->job;
    int shuffle_index = current_job->shuffle_vec_size->fetch_sub (1);
    while (shuffle_index > 0)
    {
        auto vec = (*current_job->shuffleVec)[shuffle_index - 1];
        current_job->client->reduce (vec, current_context);
        int vec_size = (int)(vec->size() * 4);
        current_job->state_counter->fetch_add(vec_size);
        shuffle_index = current_job->shuffle_vec_size->fetch_sub (1);
    }
}

/**
 * This function runs by each thread. It runs the map, shuffle and reduce operation
 * @param arg the specific thread context
 */
void *thread_function (void *arg)
{
    auto *current_context = (ThreadContext *) arg;
    map_phase (arg);
    //sort intermediate vector
    if (!current_context->intermediate_vec->empty ())
    {
        std::sort (current_context->intermediate_vec->begin (),
                   current_context->intermediate_vec->end (), comparator);
    }
    shuffle_and_barrier_phase (arg);
    reduce_phase (arg);
    return nullptr;
}

/**
 * This function starts running the MapReduce algorithm
 * @param client the given client to perform the MapReduce algorithm on
 * @param inputVec given pointer to a vector with {k1,v1} elements
 * @param outputVec  given pointer to store the algorithm output of {k3,v3}
 * elements
 * @param multiThreadLevel given num of thread to run the program on
 * @return returns the necessary information about the algorithm procedure
 */
JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel)
                             {
    JobDetails *main_job;
    auto *my_state = new (std::nothrow) JobState{UNDEFINED_STAGE, 0};
    auto *state_counter = new (std::nothrow) std::atomic<int64_t> ((inputVec.size() << 33));
    auto *threads = new (std::nothrow) pthread_t[multiThreadLevel] ();
    auto *contexts = new (std::nothrow) ThreadContext[multiThreadLevel] ();
    auto *barrier = new (std::nothrow) Barrier (multiThreadLevel);
    auto *outputvec_size = new (std::nothrow)atomic (0);
    auto *input_vec_index = new (std::nothrow) atomic (0);
    auto *shuffle_vec_size = new (std::nothrow) atomic (0);
    auto *intermediary_num_of_elements = new (std::nothrow) atomic (0);

    if (my_state == nullptr || threads == nullptr || contexts == nullptr ||
    barrier == nullptr || input_vec_index == nullptr || outputvec_size == nullptr
    || shuffle_vec_size == nullptr || state_counter == nullptr|| intermediary_num_of_elements == nullptr)

    {
        std::cout << BAD_ALLOC_ERROR << std::endl;
        exit(1);
    }
    main_job = new (std::nothrow) JobDetails{job_index++, multiThreadLevel, my_state, state_counter,
                                             &inputVec, input_vec_index, nullptr,
                                             shuffle_vec_size,
                                             &outputVec, outputvec_size,
                                             intermediary_num_of_elements, threads,
                                             contexts, barrier, &client};
    if (main_job == nullptr)
    {
        std::cout << BAD_ALLOC_ERROR << std::endl;
        exit(1);
    }
    //Initialize threads contexts
    for (int i = 0; i < multiThreadLevel; ++i)
    {
        IntermediateVec *inter_vec = new (std::nothrow) IntermediateVec ();
        if (inter_vec == nullptr)
        {
            std::cout << BAD_ALLOC_ERROR << std::endl;
            exit(1);
        }
        main_job->contexts[i] = {i, main_job, inter_vec};
        //Initialize threads
        if(pthread_create (main_job->threads + i, NULL, thread_function, main_job->contexts + i))
        {
            std::cout << PTHREAD_ERROR << std::endl;
            exit(1);
        }
    }
    return (JobHandle) main_job;
 }

 /**
  * The function saves the intermediary element in the thread context data
  * structures. In addition, the function updates the number of intermediary
  * elements using atomic counter.
  * @param key given intermediary key
  * @param value given intermediary value
  * @param context given thread context
  */
 void emit2 (K2 *key, V2 *value, void *context)
 {
     auto *current_context = (ThreadContext *) context;
     IntermediatePair pair (key, value);
     current_context->intermediate_vec->push_back (pair);
     current_context->job->intermediary_num_of_elements->fetch_add(1);
 }

 /**
 * The function saves the intermediary element in the thread context data
 * structures. In addition, the function updates the number of intermediary
 * elements using atomic counter.
 * @param key given intermediary key
 * @param value given intermediary value
 * @param context given thread context
  * */
 void emit3 (K3 *key, V3 *value, void *context)
 {
     auto *current_context = (ThreadContext *) context;
     OutputPair pair (key, value);
     current_context->job->protect_output.lock ();
     current_context->job->outputVec->push_back (pair);
     current_context->job->outputvec_size->fetch_add(1);
     current_context->job->protect_output.unlock ();
 }

 /**
  * Waits until the given JobHandle is done
  * @param job given job to wait for
  */
 void waitForJob (JobHandle job)
 {
     auto *current_job = (JobDetails *) job;
     current_job->protect_join.lock ();
     if (!current_job->wait_flag)
     {
         for (int i = 0; i < current_job->num_of_threads; ++i)
         {
             if(pthread_join (current_job->threads[i], NULL))
             {
                 std::cout<<PTHREAD_JOIN_ERROR<<std::endl;
                 exit(1);
             }
         }
         current_job->wait_flag = true;
     }
     current_job->protect_join.unlock ();
 }

 /**
  * Releasing all resources of a job. You should prevent releasing resources
  * before the job finished. After this function is called the job handle will be invalid.
  * In case that the function is called and the job is not finished yet wait until the job is
  * finished to close it.
  * @param job
  */
 void closeJobHandle (JobHandle job)
 {
     auto *current_job = (JobDetails *) job;
     update_percentage(job);
     waitForJob(job);
     delete current_job->state;
     delete current_job->state_counter;
     delete current_job->input_vec_index;
     for (auto vec: *current_job->shuffleVec)
     {
         vec->clear ();
         delete vec;
     }
     delete current_job->shuffleVec;
     delete current_job->shuffle_vec_size;
     delete current_job->outputvec_size;
     delete[] current_job->threads;
     for (int i = 0; i < current_job->num_of_threads; i++)
     {
         current_job->contexts[i].intermediate_vec->clear ();
         delete current_job->contexts[i].intermediate_vec;
     }
     delete[] current_job->contexts;
     delete current_job->intermediary_num_of_elements;
     delete current_job->barrier;
     delete current_job;
 }

 /**
  * This function gets a JobHandle and updates the state of the job into the
  * given JobState struct
  * @param job the given job state
  * @param state the given job state to update
  */
 void getJobState (JobHandle job, JobState *state)
 {
     auto *current_job = (JobDetails *) job;
     update_percentage(current_job);
     state->stage = current_job->state->stage;
     state->percentage = current_job->state->percentage;
 }