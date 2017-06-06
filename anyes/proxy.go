package anyes

import (
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"net"
	"time"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/gobplexer"
)

func init() {
	gob.Register(&packet{})
}

const (
	keepaliveInterval = time.Minute
	keepaliveMaxDelay = time.Minute * 2
)

type packetType int

const (
	packetInit packetType = iota
	packetRun
	packetUpdate
)

type packet struct {
	Type packetType

	// Used for init requests.
	InitSeed  int64
	InitSize  int
	InitModel []byte

	// Used for run requests.
	Stop  *StopConds
	Scale float64
	Seed  int64

	// Used for run responses.
	Rollout *Rollout

	// Used for update requests.
	Scales []float64
	Seeds  []int64

	// Used for all responses.
	Err *string
}

func newPacketErr(err error) *packet {
	if err == nil {
		return &packet{}
	}
	s := err.Error()
	return &packet{Err: &s}
}

// ProxyProvide provides a Slave to the other end of a
// proxy, which should be using ProxyConsume.
//
// This blocks until the proxy connection ends.
// It automatically closes c.
func ProxyProvide(c io.ReadWriteCloser, s Slave) (err error) {
	defer essentials.AddCtxTo("provide proxy", &err)

	rootConn := gobplexer.NetConnection(c)
	defer rootConn.Close()

	connector := gobplexer.MultiplexConnector(rootConn)
	defer connector.Close()

	conn, err := gobplexer.KeepaliveConnector(connector, keepaliveInterval,
		keepaliveMaxDelay)
	if err != nil {
		return err
	}
	defer conn.Close()

	for {
		p, err := receivePacket(conn)
		if err != nil {
			return err
		}
		switch p.Type {
		case packetInit:
			err := s.Init(p.InitModel, p.InitSeed, p.InitSize)
			err = conn.Send(newPacketErr(err))
			if err != nil {
				return err
			}
		case packetRun:
			rollout, err := s.Run(p.Stop, p.Scale, p.Seed)
			resP := newPacketErr(err)
			resP.Rollout = rollout
			err = conn.Send(resP)
			if err != nil {
				return err
			}
		case packetUpdate:
			err := s.Update(p.Scales, p.Seeds)
			err = conn.Send(newPacketErr(err))
			if err != nil {
				return err
			}
		default:
			return fmt.Errorf("unknown packet type: %v", p.Type)
		}
	}
}

// SlaveProxy is a connection to a remote Slave.
//
// A SlaveProxy should be closed to clean up resources
// associated with it.
type SlaveProxy interface {
	io.Closer
	Slave
}

type slaveProxy struct {
	closers []io.Closer
	conn    gobplexer.Connection
}

// ProxyConsume connects to a Slave proxy which is running
// ProxyProvide on the other end.
func ProxyConsume(c io.ReadWriteCloser) (slave SlaveProxy, err error) {
	defer essentials.AddCtxTo("consume proxy", &err)

	res := &slaveProxy{}

	rootConn := gobplexer.NetConnection(c)
	res.closers = append(res.closers, rootConn)

	listener := gobplexer.MultiplexListener(rootConn)
	res.closers = append(res.closers, listener)

	conn, err := gobplexer.KeepaliveListener(listener, keepaliveInterval,
		keepaliveMaxDelay)
	if err != nil {
		res.Close()
		return nil, err
	}
	res.closers = append(res.closers, conn)
	res.conn = conn

	return res, nil
}

func (r *slaveProxy) Close() error {
	for _, c := range r.closers {
		c.Close()
	}
	return nil
}

func (s *slaveProxy) Init(data []byte, seed int64, size int) (err error) {
	defer essentials.AddCtxTo("slave proxy init", &err)
	return s.call(&packet{
		Type:      packetInit,
		InitModel: data,
		InitSeed:  seed,
		InitSize:  size,
	})
}

func (s *slaveProxy) Run(sc *StopConds, scale float64, seed int64) (r *Rollout,
	err error) {
	defer essentials.AddCtxTo("slave proxy run", &err)
	p := &packet{
		Type:  packetRun,
		Stop:  sc,
		Scale: scale,
		Seed:  seed,
	}
	if err := s.conn.Send(p); err != nil {
		return nil, err
	}
	p, err = receivePacket(s.conn)
	if err != nil {
		return nil, err
	}
	if p.Err != nil {
		return p.Rollout, errors.New(*p.Err)
	}
	return p.Rollout, nil
}

func (s *slaveProxy) Update(scales []float64, seeds []int64) (err error) {
	defer essentials.AddCtxTo("slave proxy update", &err)
	return s.call(&packet{
		Type:   packetUpdate,
		Scales: scales,
		Seeds:  seeds,
	})
}

func (s *slaveProxy) call(p *packet) error {
	if err := s.conn.Send(p); err != nil {
		return err
	}
	p, err := receivePacket(s.conn)
	if err != nil {
		return err
	}
	if p.Err != nil {
		return errors.New(*p.Err)
	}
	return nil
}

func receivePacket(c gobplexer.Connection) (*packet, error) {
	packetObj, err := c.Receive()
	if err != nil {
		return nil, err
	}
	packet, ok := packetObj.(*packet)
	if !ok {
		return nil, fmt.Errorf("bad packet type: %T", packetObj)
	}
	return packet, nil
}

// ProxyListen listens for incomming connections on l and
// wraps each connection with ProxyConsume.
// The resulting Slaves are then added to m.
//
// If loger is non-nil, it is passed messages whenever a
// a slave connects or fails to connect with an error.
//
// This blocks until l.Accept returns with an error, at
// which point the error is returned.
//
// In order to properly cleanup Slaves, you should close
// slaves which implement SlaveProxy in m.SlaveError.
// You should also close all remaining slaves after you
// are done with m.
func ProxyListen(l net.Listener, m *Master,
	logger func(msg ...interface{})) (err error) {
	defer essentials.AddCtxTo("ProxyListen", &err)
	for {
		conn, err := l.Accept()
		if err != nil {
			return err
		}
		go func(conn net.Conn) {
			sendLog := func(str string) {
				if logger != nil {
					logger(conn.RemoteAddr().String() + ": " + str)
				}
			}
			sendLog("new connection")
			slave, err := ProxyConsume(conn)
			if err != nil {
				sendLog(err.Error())
				return
			}
			if err := m.AddSlave(slave); err != nil {
				slave.Close()
				sendLog(err.Error())
				return
			}
			sendLog("slave added")
		}(conn)
	}
}
