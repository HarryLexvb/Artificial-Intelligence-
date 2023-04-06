//
// Created by HarryLex on 5/04/2023.
//

#ifndef ASTAR_BESTFISRT_ALGORITHMFACTORY_H
#define ASTAR_BESTFISRT_ALGORITHMFACTORY_H

#include "Algorithm.h"
#include "AStar.h"
#include "BestFS.h"


class AlgorithmFactory {
public:
    static std::unique_ptr<Algorithm> generateAlgorithm(const std::string_view& graphtype ,Graph& graph) {
        if (graphtype == "AStar") {
            return std::make_unique<AStar>(graph);
        }
        else if (graphtype == "BestFS") {
            return std::make_unique<BestFS>(graph);
        }
        else return nullptr;
    }

};

#endif //ASTAR_BESTFISRT_ALGORITHMFACTORY_H
