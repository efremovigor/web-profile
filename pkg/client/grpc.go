package client

import (
	"context"
	"fmt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"time"
	pb "web_profile/internal/generated/image_search"
)

func CreateGRPCClient(config GRPCClientConfig) (pb.ImageSearchServiceClient, func(), error) {
	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx,
		fmt.Sprintf("%s:%d", config.Host, config.Port),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC connection failed: %v", err)
	}

	client := pb.NewImageSearchServiceClient(conn)
	return client, func() { conn.Close() }, nil
}

type ImageSearchRequest struct {
	TopK    int  `form:"top_k" json:"top_k" binding:"min=1,max=100"`
	NoCache bool `form:"no_cache" json:"no_cache"`
}

type GRPCClientConfig struct {
	Host    string
	Port    int
	Timeout time.Duration
}
