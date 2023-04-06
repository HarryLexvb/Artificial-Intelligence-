//
// Created by HarryLex on 5/04/2023.
//

#ifndef ASTAR_BESTFISRT_LOCATION_H
#define ASTAR_BESTFISRT_LOCATION_H

struct Location {
    int posx;
    int posy;
    bool operator==(const Location& other) const{
        return posx == other.posx && posy == other.posy;
    }
};


#endif //ASTAR_BESTFISRT_LOCATION_H
