cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr)
project(subcontrol)

target_sources(subcontrol PRIVATE src/main.cpp)