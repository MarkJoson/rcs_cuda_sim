#ifndef __MAP_GENERATOR__
#define __MAP_GENERATOR__

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <random>

namespace map_gen
{

class Room
{
public:
    int x1, y1, x2, y2;

    Room(int x, int y, int w, int h)
        : x1(x), y1(y), x2(x + w), y2(y + h) {}

    std::pair<int, int> center() const
    {
        return {(x1 + x2) / 2, (y1 + y2) / 2};
    }

    bool intersect(const Room &other) const
    {
        return (x1 <= other.x2 && x2 >= other.x1 &&
                y1 <= other.y2 && y2 >= other.y1);
    }
};

class Leaf
{
public:
    int x, y, w, h;
    static const int MIN_LEAF_SIZE = 10;
    std::unique_ptr<Leaf> child1;
    std::unique_ptr<Leaf> child2;
    std::unique_ptr<Room> room;

    Leaf(int _x, int _y, int width, int height) : x(_x), y(_y), w(width), h(height) {}

    bool split();
};

class DungeonGenerator
{
protected:
    int width_;
    int height_;
    std::vector<std::vector<int>> map_;

public:
    DungeonGenerator(int width, int height)
        : width_(width), height_(height), map_(width, std::vector<int>(height, 1)) {}
    
    int getWidth() { return width_; }
    int getHeight() { return height_; }

    virtual void generate() = 0;
    virtual DungeonGenerator *clone() const = 0;
    const std::vector<std::vector<int>> &getMap() const { return map_; }

protected:
    void clearMap()
    {
        for (auto &row : map_)
        {
            std::fill(row.begin(), row.end(), 1);
        }
    }
};

class TunnelingGenerator : public DungeonGenerator
{
private:
    const int ROOM_MAX_SIZE = 15;
    const int ROOM_MIN_SIZE = 6;
    const int MAX_ROOMS = 30;
    std::vector<Room> rooms_;

public:
    using DungeonGenerator::DungeonGenerator;
    DungeonGenerator *clone() const override { return new TunnelingGenerator(*this); }
    void generate() override;

private:
    void createRoom(const Room &room);
    void createHorizontalTunnel(int x1, int x2, int y);
    void createVerticalTunnel(int y1, int y2, int x);
};

class BSPGenerator : public DungeonGenerator
{
private:
    const int MAX_LEAF_SIZE = 24;
    const int ROOM_MAX_SIZE = 15;
    const int ROOM_MIN_SIZE = 6;
    std::vector<Leaf *> leafs_;

public:
    using DungeonGenerator::DungeonGenerator;
    DungeonGenerator *clone() const override { return new BSPGenerator(*this); }
    void generate() override;

private:
    void createRooms();
    void createRoom(const Room &room);
    void connectRooms(const Room &room1, const Room &room2);
    void createHorizontalTunnel(int x1, int x2, int y);
    void createVerticalTunnel(int y1, int y2, int x);
};

class DrunkardWalkGenerator : public DungeonGenerator
{
private:
    const float PERCENT_GOAL = 0.4f;
    const int WALK_ITERATIONS = 25000;
    const float WEIGHTED_TOWARD_CENTER = 0.15f;
    const float WEIGHTED_TOWARD_PREVIOUS_DIRECTION = 0.7f;
    int filled_;
    std::string previousDirection_;
    int drunkardX_, drunkardY_;

public:
    using DungeonGenerator::DungeonGenerator;
    DungeonGenerator *clone() const override { return new DrunkardWalkGenerator(*this); }

    void generate() override;

private:
    void walk();
};

class CellularAutomataGenerator : public DungeonGenerator
{
private:
    const int ITERATIONS = 50000;
    const int NEIGHBORS = 4;
    const float WALL_PROBABILITY = 0.5f;
    const int ROOM_MIN_SIZE = 16;
    const int ROOM_MAX_SIZE = 500;
    const bool SMOOTH_EDGES = true;
    const int SMOOTHING = 1;
    std::vector<std::set<std::pair<int, int>>> caves_;

public:
    using DungeonGenerator::DungeonGenerator;
    CellularAutomataGenerator *clone() const override { return new CellularAutomataGenerator(*this); }
    void generate() override;

private:
    void randomFillMap();
    void createCaves();
    void cleanUpMap();
    void createTunnel(const std::set<std::pair<int, int>> &cave1, const std::pair<int, int> &point1, const std::pair<int, int> &point2);
    int getAdjacentWallsSimple(int x, int y);
    int getAdjacentWalls(int x, int y);
    void getCaves();
    void floodFill(int startX, int startY);
    void connectCaves();
    bool checkConnectivity(const std::set<std::pair<int, int>> &cave1, const std::set<std::pair<int, int>> &cave2);
    double distanceFormula(const std::pair<int, int> &p1, const std::pair<int, int> &p2);
};

class RoomAdditionGenerator : public DungeonGenerator
{
private:
    const int ROOM_MAX_SIZE = 18;
    const int ROOM_MIN_SIZE = 16;
    const int MAX_NUM_ROOMS = 30;
    const int SQUARE_ROOM_MAX_SIZE = 12;
    const int SQUARE_ROOM_MIN_SIZE = 6;
    const int CROSS_ROOM_MAX_SIZE = 12;
    const int CROSS_ROOM_MIN_SIZE = 6;
    const float CAVERN_CHANCE = 0.4f;
    const int CAVERN_MAX_SIZE = 35;
    const float WALL_PROBABILITY = 0.45f;
    const int NEIGHBORS = 4;
    const float SQUARE_ROOM_CHANCE = 0.2f;
    const float CROSS_ROOM_CHANCE = 0.15f;
    const int BUILD_ROOM_ATTEMPTS = 500;
    const int PLACE_ROOM_ATTEMPTS = 20;
    const int MAX_TUNNEL_LENGTH = 12;

    std::vector<std::vector<int>> currentRoom_;
    std::vector<std::vector<std::vector<int>>> rooms_;

public:
    using DungeonGenerator::DungeonGenerator;
    DungeonGenerator *clone() const override { return new RoomAdditionGenerator(*this); }
    void generate() override;

private:
    std::vector<std::vector<int>> generateRoom();
    std::vector<std::vector<int>> generateSquareRoom();
    std::vector<std::vector<int>> generateCrossRoom();
    std::vector<std::vector<int>> generateCellularRoom();
    std::vector<std::vector<int>> generateCavernRoom();
    void addRoom(int roomX, int roomY, const std::vector<std::vector<int>> &room);
    void addTunnel(std::pair<int, int> wallTile, std::pair<int, int> direction, int tunnelLength);
    std::pair<int, int> getDirection();
    std::tuple<int, int, std::pair<int, int>, std::pair<int, int>, int> placeRoom(
        const std::vector<std::vector<int>> &room);
    bool canPlaceRoom(const std::vector<std::vector<int>> &room, int roomX, int roomY);
    std::pair<int, int> getRoomDimensions(const std::vector<std::vector<int>> &room);
    int countAdjacentWalls(int x, int y, const std::vector<std::vector<int>> &room);
};

class CityWallsGenerator : public DungeonGenerator
{
private:
    const int MAX_LEAF_SIZE = 30;
    const int ROOM_MAX_SIZE = 16;
    const int ROOM_MIN_SIZE = 8;
    std::vector<Leaf *> leafs_;
    std::vector<Room> rooms_;

public:
    using DungeonGenerator::DungeonGenerator;
    DungeonGenerator *clone() const override { return new CityWallsGenerator(*this); }
    void generate() override;

private:
    void createRooms();
    void createRoom(const Room &room);
    void createDoors();
};

class MazeWithRoomsGenerator : public DungeonGenerator
{
private:
    const int ROOM_MAX_SIZE = 13;
    const int ROOM_MIN_SIZE = 6;
    const int BUILD_ROOM_ATTEMPTS = 100;
    const float CONNECTION_CHANCE = 0.04f;
    const float WINDING_PERCENT = 0.1f;
    const bool ALLOW_DEAD_ENDS = false;

    std::vector<std::vector<int>> regions_;
    int currentRegion_;

public:
    using DungeonGenerator::DungeonGenerator;
    DungeonGenerator *clone() const override { return new MazeWithRoomsGenerator(*this); }
    void generate() override;

private:
    void addRooms(int mapWidth, int mapHeight);
    void growMaze(std::pair<int, int> start, int mapWidth, int mapHeight);
    bool canCarve(std::pair<int, int> pos, std::pair<int, int> dir, int mapWidth, int mapHeight);
    void connectRegions(int mapWidth, int mapHeight);
    void removeDeadEnds(int mapWidth, int mapHeight);
    void startNewRegion();
    void carve(int x, int y);
    void createRoom(const Room &room);
};

class MessyBSPGenerator : public DungeonGenerator
{
private:
    const int MAX_LEAF_SIZE = 24;
    const int ROOM_MAX_SIZE = 15;
    const int ROOM_MIN_SIZE = 6;
    const bool SMOOTH_EDGES = true;
    const int SMOOTHING = 1;
    const int FILLING = 3;

    std::vector<Leaf *> leafs_;

public:
    using DungeonGenerator::DungeonGenerator;
    DungeonGenerator *clone() const override { return new MessyBSPGenerator(*this); }
    void generate() override;

private:
    void createRooms(Leaf &leaf);
    void createRoom(const Room &room);
    void connectRooms(const Room &room1, const Room &room2);
    void smoothMap();
    int countAdjacentWalls(int x, int y);
};

} // namespace MapGen



#endif //__MAP_GENERATOR__
