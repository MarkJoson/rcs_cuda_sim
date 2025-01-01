#include <algorithm>
#include <boost/graph/graph_selectors.hpp>
#include <iostream>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>

using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS>;

template<typename Graph>
struct exercise_graph {
    exercise_graph(Graph& g) : g_(g) {}
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;

    void operator()(const Vertex& v) {
        using IndexType = boost::property_map<Graph, boost::vertex_index_t>::type;
        IndexType index;
        index = boost::get(boost::vertex_index, g_);

        typename boost::graph_traits<Graph>::in_edge_iterator in_i, in_end;
        typename boost::graph_traits<Graph>::out_edge_iterator out_i, out_end;
        typename boost::graph_traits<Graph>::adjacency_iterator ai, ai_end;

        std::cout << "vertex: " << index[v] << std::endl;

        // 遍历所有入边
        for(std::tie(in_i, in_end) = boost::in_edges(v, g_); in_i != in_end; in_i++) {
            std::cout << "in edge from " << index[boost::source(*in_i, g_)] << " to " << index[boost::target(*in_i, g_)] << std::endl;
        }

        // 遍历所有出边
        for(std::tie(out_i, out_end) = boost::out_edges(v, g_); out_i != out_end; out_i++) {
            std::cout << "out edge from " << index[boost::source(*out_i, g_)] << " to " << index[boost::target(*out_i, g_)] << std::endl;
        }

        // 遍历所有邻接点
        for(std::tie(ai, ai_end) = boost::adjacent_vertices(v, g_); ai != ai_end; ai++) {
            std::cout << "adjacent vertex " << index[*ai] << std::endl;
        }

    }
    Graph& g_;
};


int main(int argc, char** args) {
    enum {A,B,C,D,E, NUM_VERTICES};

    Graph g(NUM_VERTICES);

    std::pair<int, int> edges[] = {
        {A,B}, {A,D}, {C,A}, {D,C}, {C,E}, {B,D}, {D,E}
    };

    const int num_edges = sizeof(edges) / sizeof(edges[0]);

    for(auto &edge : edges) {
        boost::add_edge(edge.first, edge.second, g);
    }

    // 将顶点索引映射到顶点属性（仍然是顶点索引）
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = boost::get(boost::vertex_index, g);

    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    std::pair<vertex_iter, vertex_iter> vp;

    for(vp=boost::vertices(g); vp.first != vp.second; ++vp.first) {
        std::cout << "index: " << index[*vp.first] << std::endl;
    }

    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
    edge_iter ei, ei_end;
    for(std::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ei++) {
        std::cout << "Edge from:" << index[boost::source(*ei, g)] << ", to:" << index[boost::source(*ei, g)] << std::endl;
    }

    for (int i = 0; i < num_edges; i++) {
        boost::add_edge(edges[i].first, edges[i].second, g);
    }

    vp = boost::vertices(g);
    std::for_each(vp.first, vp.second, exercise_graph<Graph>(g));

    

    return 0;

}