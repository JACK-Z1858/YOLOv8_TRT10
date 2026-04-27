#pragma once

#include "thread/thread.hpp"

void Producer(threadSafeQueue& p2c);

void Consumer(threadSafeQueue& p2c);